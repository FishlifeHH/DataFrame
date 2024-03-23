#include <DataFrame/DataFrame.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "async/scoped_inline_task.hpp"
#include "cache/accessor.hpp"
#include "option.hpp"
#include "rdma/client.hpp"
#include "utils/control.hpp"
#include "utils/debug.hpp"
#include "utils/parallel.hpp"
#include "utils/perf.hpp"
// #define STANDALONE
// simple: ~74M, full: ~16G
// #define SIMPLE_BENCH

#ifdef STANDALONE
#include "rdma/server.hpp"
#endif

using namespace FarLib;
using namespace FarLib::rdma;
using namespace FarLib::cache;
using namespace std::chrono_literals;
using namespace hmdf;

// Download dataset at https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page.
// The following code is implemented based on the format of 2016 datasets.
static double haversine(double lat1, double lon1, double lat2, double lon2)
{
    // Distance between latitudes and longitudes
    double dLat = (lat2 - lat1) * M_PI / 180.0;
    double dLon = (lon2 - lon1) * M_PI / 180.0;

    // Convert to radians.
    lat1 = lat1 * M_PI / 180.0;
    lat2 = lat2 * M_PI / 180.0;

    // Apply formulae.
    double a   = pow(sin(dLat / 2), 2) + pow(sin(dLon / 2), 2) * cos(lat1) * cos(lat2);
    double rad = 6371;
    double c   = 2 * asin(sqrt(a));
    return rad * c;
}

StdDataFrame<uint64_t> load_data()
{
#ifdef SIMPLE_BENCH
    // const char* file_path =
    // "/home/huanghong/mem_parallel/motivation/FarLib/build/very_simple.csv";
    const char* file_path = "/mnt/simple.csv";

#else
    const char* file_path = "/mnt/all.csv";
#endif
    return read_csv<-1, int, SimpleTime, SimpleTime, int, double, double, double, int, char, double,
                    double, int, double, double, double, double, double, double, double>(
        file_path, "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
        "trip_distance", "pickup_longitude", "pickup_latitude", "RatecodeID", "store_and_fwd_flag",
        "dropoff_longitude", "dropoff_latitude", "payment_type", "fare_amount", "extra", "mta_tax",
        "tip_amount", "tolls_amount", "improvement_surcharge", "total_amount");
}

template <Algorithm alg = DEFAULT_ALG>
void print_number_vendor_ids_and_unique(StdDataFrame<uint64_t>& df)
{
    std::cout << "print_number_vendor_ids_and_unique()" << std::endl;
    std::cout << "Number of vendor_ids in the train dataset: "
              << df.get_column<int>("VendorID").size() << std::endl;
    std::cout << "Number of unique vendor_ids in the train dataset:"
              << df.get_col_unique_values<alg, int>("VendorID").size() << std::endl;
    std::cout << std::endl;
}

template <Algorithm alg = DEFAULT_ALG>
void print_passage_counts_by_vendor_id(StdDataFrame<uint64_t>& df, int vendor_id)
{
    std::cout << "print_passage_counts_by_vendor_id(vendor_id), vendor_id = " << vendor_id
              << std::endl;

    auto sel_vendor_functor = [&](const uint64_t&, const int& vid) -> bool {
        return vid == vendor_id;
    };
    auto start = get_cycles();
    decltype(df.get_data_by_sel<alg, int, decltype(sel_vendor_functor), int, SimpleTime, double,
                                char>("VendorID", sel_vendor_functor)) sel_df;
    // perf_profile([&]() {
    sel_df =
        df.get_data_by_sel<alg, int, decltype(sel_vendor_functor), int, SimpleTime, double, char>(
            "VendorID", sel_vendor_functor);
    // }).print();
    auto end = get_cycles();
    std::cout << "sel df get: " << end - start << std::endl;
    auto& passage_count_vec = sel_df.template get_column<int>("passenger_count");
    std::map<int, int> passage_count_map;
    start = get_cycles();
    if constexpr (alg == DEFAULT) {
        for (auto passage_count : passage_count_vec) {
            passage_count_map[passage_count]++;
        }
    } else if constexpr (alg == UTHREAD) {
        uthread::parallel_for_with_scope<1>(
            uthread::get_worker_count() * UTH_FACTOR, passage_count_vec.size(),
            [&](size_t i, DereferenceScope& scope) {
                ON_MISS_BEGIN
                uthread::yield();
                ON_MISS_END
                auto passage_count = *(passage_count_vec.at(i, scope, __on_miss__));
                passage_count_map[passage_count]++;
            });
    } else if constexpr (alg == PREFETCH) {
        RootDereferenceScope scope;
        auto it         = passage_count_vec.clbegin(scope);
        size_t vec_size = passage_count_vec.size();
        for (size_t i = 0; i < vec_size; i++, it.next(scope)) {
            passage_count_map[*it]++;
        }
    } else if constexpr (alg == PARAROUTINE) {
        struct Context {
            FarLib::FarVector<int>* passage_count_vec;
            std::map<int, int>* passage_count_map;
            size_t idx;
            size_t idx_end;
            decltype(passage_count_vec->clbegin()) it;
            bool fetch_end;
            Context(FarLib::FarVector<int>* passage_count_vec,
                    std::map<int, int>* passage_count_map, size_t idx, size_t idx_end)
                : passage_count_vec(passage_count_vec),
                  passage_count_map(passage_count_map),
                  idx(idx),
                  idx_end(idx_end),
                  fetch_end(false)
            {
            }

            void pin()
            {
                it.pin();
            }

            void unpin()
            {
                it.unpin();
            }

            bool fetched()
            {
                return it.at_local();
            }

            bool run(DereferenceScope& scope)
            {
                if (fetch_end) {
                    goto next;
                }
                while (idx < idx_end) {
                    it        = passage_count_vec->clbegin().nextn(idx);
                    fetch_end = true;
                    if (!it.async_fetch(scope)) {
                        return false;
                    }
                next:
                    (*passage_count_map)[*it]++;
                    idx++;
                    fetch_end = false;
                }
                return true;
            }
        };
        RootDereferenceScope scope;
        const size_t vec_size = passage_count_vec.size();
        const size_t block    = (vec_size + CTX_COUNT - 1) / CTX_COUNT;
        SCOPED_INLINE_ASYNC_FOR(Context, size_t, i, 0, i < vec_size, i += block, scope)
        return Context(&passage_count_vec, &passage_count_map, i, std::min(i + block, vec_size));
        SCOPED_INLINE_ASYNC_FOR_END
    } else {
        ERROR("alg dont exist");
    }
    end = get_cycles();
    std::cout << "map fill: " << end - start << std::endl;
    for (auto& [passage_count, cnt] : passage_count_map) {
        std::cout << "passage_count= " << passage_count << ", cnt = " << cnt << std::endl;
    }
    std::cout << std::endl;
}

template <Algorithm alg = DEFAULT_ALG>
void calculate_trip_duration(StdDataFrame<uint64_t>& df)
{
    std::cout << "calculate_trip_duration()" << std::endl;

    auto& pickup_time_vec  = df.get_column<SimpleTime>("tpep_pickup_datetime");
    auto& dropoff_time_vec = df.get_column<SimpleTime>("tpep_dropoff_datetime");
    assert(pickup_time_vec.size() == dropoff_time_vec.size());

    auto start = get_cycles();
    FarLib::FarVector<uint64_t> duration_vec(pickup_time_vec.size());
    if constexpr (alg == DEFAULT) {
        for (uint64_t i = 0; i < pickup_time_vec.size(); i++) {
            auto pickup_time_second  = pickup_time_vec[i]->to_second();
            auto dropoff_time_second = dropoff_time_vec[i]->to_second();
            *duration_vec[i]         = (dropoff_time_second - pickup_time_second);
        }
    } else if constexpr (alg == UTHREAD) {
        uthread::parallel_for_with_scope<1>(
            uthread::get_worker_count() * UTH_FACTOR, pickup_time_vec.size(),
            [&](size_t i, DereferenceScope& scope) {
                ON_MISS_BEGIN
                uthread::yield();
                ON_MISS_END
                auto pickup_time_second = (pickup_time_vec.at(i, scope, __on_miss__))->to_second();
                auto dropoff_time_second =
                    (dropoff_time_vec.at(i, scope, __on_miss__))->to_second();
                *(duration_vec.at_mut(i, scope, __on_miss__)) =
                    dropoff_time_second - pickup_time_second;
            });
    } else if constexpr (alg == PREFETCH) {
        struct Scope : public RootDereferenceScope {
            decltype(pickup_time_vec.clbegin()) pickup_time_it;
            decltype(dropoff_time_vec.clbegin()) dropoff_time_it;
            decltype(duration_vec.lbegin()) duration_it;
            void pin() const override
            {
                pickup_time_it.pin();
                dropoff_time_it.pin();
                duration_it.pin();
            }
            void unpin() const override
            {
                pickup_time_it.unpin();
                dropoff_time_it.unpin();
                duration_it.unpin();
            }

            void next()
            {
                pickup_time_it.next(*this);
                dropoff_time_it.next(*this);
                duration_it.next(*this);
            }
        } scope;
        scope.pickup_time_it  = pickup_time_vec.clbegin(scope);
        scope.dropoff_time_it = dropoff_time_vec.clbegin(scope);
        scope.duration_it     = duration_vec.lbegin(scope);
        for (size_t i = 0; i < pickup_time_vec.size(); i++, scope.next()) {
            *(scope.duration_it) =
                (scope.dropoff_time_it)->to_second() - (scope.pickup_time_it)->to_second();
        }
    } else if constexpr (alg == PARAROUTINE) {
        using pickup_time_vec_t  = std::remove_reference_t<decltype(pickup_time_vec)>;
        using dropoff_time_vec_t = std::remove_reference_t<decltype(dropoff_time_vec)>;
        using duration_vec_t     = std::remove_reference_t<decltype(duration_vec)>;
        struct Context {
            pickup_time_vec_t* pickup_time_vec;
            dropoff_time_vec_t* dropoff_time_vec;
            duration_vec_t* duration_vec;
            decltype(pickup_time_vec->clbegin()) pickup_time_it;
            decltype(dropoff_time_vec->clbegin()) dropoff_time_it;
            decltype(duration_vec->lbegin()) duration_it;
            size_t idx;
            size_t idx_end;
            bool fetch_end;

            Context(pickup_time_vec_t* pickup_time_vec, dropoff_time_vec_t* dropoff_time_vec,
                    duration_vec_t* duration_vec, size_t idx, size_t idx_end)
                : pickup_time_vec(pickup_time_vec),
                  dropoff_time_vec(dropoff_time_vec),
                  duration_vec(duration_vec),
                  idx(idx),
                  idx_end(idx_end),
                  fetch_end(false)
            {
            }

            void pin()
            {
                pickup_time_it.pin();
                dropoff_time_it.pin();
                duration_it.pin();
            }

            void unpin()
            {
                pickup_time_it.unpin();
                dropoff_time_it.unpin();
                duration_it.unpin();
            }

            bool fetched()
            {
                return duration_it.at_local() && dropoff_time_it.at_local() &&
                       pickup_time_it.at_local();
            }

            bool run(DereferenceScope& scope)
            {
                if (fetch_end) {
                    goto next;
                }
                while (idx < idx_end) {
                    pickup_time_it = pickup_time_vec->clbegin().nextn(idx);
                    pickup_time_it.async_fetch(scope);
                    dropoff_time_it = dropoff_time_vec->clbegin().nextn(idx);
                    dropoff_time_it.async_fetch(scope);
                    duration_it = duration_vec->lbegin().nextn(idx);
                    duration_it.async_fetch(scope);
                    fetch_end = true;
                    if (!fetched()) {
                        return false;
                    }
                next:
                    *duration_it = dropoff_time_it->to_second() - pickup_time_it->to_second();
                    idx++;
                    fetch_end = false;
                }
                return true;
            }
        };
        RootDereferenceScope scope;
        const size_t block = (pickup_time_vec.size() + CTX_COUNT - 1) / CTX_COUNT;
        SCOPED_INLINE_ASYNC_FOR(Context, size_t, i, 0, i < pickup_time_vec.size(), i += block,
                                scope)
        return Context(&pickup_time_vec, &dropoff_time_vec, &duration_vec, i,
                       std::min(i + block, pickup_time_vec.size()));
        SCOPED_INLINE_ASYNC_FOR_END
    } else {
        ERROR("algorithm dont exist");
    }
    auto end = get_cycles();
    std::cout << "duration cal: " << end - start << std::endl;

    start = get_cycles();
    df.load_column<alg>("duration", std::move(duration_vec), nan_policy::dont_pad_with_nans);
    end = get_cycles();
    std::cout << "duration load: " << end - start << std::endl;
    start = get_cycles();
    MaxVisitor<uint64_t> max_visitor;
    MinVisitor<uint64_t> min_visitor;
    MeanVisitor<uint64_t> mean_visitor;
    df.multi_visit<alg>(std::make_pair("duration", &max_visitor),
                        std::make_pair("duration", &min_visitor),
                        std::make_pair("duration", &mean_visitor));
    end = get_cycles();
    std::cout << "duration visit: " << end - start << std::endl;
    std::cout << "Mean duration = " << mean_visitor.get_result() << " seconds" << std::endl;
    std::cout << "Min duration = " << min_visitor.get_result() << " seconds" << std::endl;
    std::cout << "Max duration = " << max_visitor.get_result() << " seconds" << std::endl;
    std::cout << std::endl;
}

template <Algorithm alg = DEFAULT_ALG>
void calculate_distribution_store_and_fwd_flag(StdDataFrame<uint64_t>& df)
{
    std::cout << "calculate_distribution_store_and_fwd_flag()" << std::endl;

    auto start              = get_cycles();
    auto sel_N_saff_functor = [&](const uint64_t&, const char& saff) -> bool {
        return saff == 'N';
    };
    auto N_df =
        df.get_data_by_sel<alg, char, decltype(sel_N_saff_functor), int, SimpleTime, double, char>(
            "store_and_fwd_flag", sel_N_saff_functor);
    std::cout << static_cast<double>(N_df.get_index().size()) / df.get_index().size() << std::endl;
    auto end = get_cycles();
    std::cout << "N-df get : " << end - start << std::endl;
    start                   = get_cycles();
    auto sel_Y_saff_functor = [&](const uint64_t&, const char& saff) -> bool {
        return saff == 'Y';
    };
    auto Y_df =
        df.get_data_by_sel<alg, char, decltype(sel_Y_saff_functor), int, SimpleTime, double, char>(
            "store_and_fwd_flag", sel_Y_saff_functor);
    end = get_cycles();
    std::cout << "Y-df get: " << end - start << std::endl;
    start                     = get_cycles();
    auto unique_vendor_id_vec = Y_df.template get_col_unique_values<alg, int>("VendorID");
    end                       = get_cycles();
    std::cout << "unique get: " << end - start << std::endl;
    std::cout << '{';
    if constexpr (alg == DEFAULT) {
        for (auto& vector_id : unique_vendor_id_vec) {
            std::cout << vector_id << ", ";
        }
    } else if constexpr (alg == UTHREAD) {
        uthread::parallel_for_with_scope<1>(
            uthread::get_worker_count() * UTH_FACTOR, unique_vendor_id_vec.size(),
            [&](size_t i, DereferenceScope& scope) {
                ON_MISS_BEGIN
                uthread::yield();
                ON_MISS_END
                std::cout << *(unique_vendor_id_vec.at(i, scope, __on_miss__)) << ", ";
            });
    } else if constexpr (alg == PREFETCH) {
        RootDereferenceScope scope;
        auto it         = unique_vendor_id_vec.clbegin(scope);
        size_t vec_size = unique_vendor_id_vec.size();
        for (size_t i = 0; i < vec_size; i++, it.next(scope)) {
            std::cout << *it << ", ";
        }
    } else if constexpr (alg == PARAROUTINE) {
        using unique_vendor_id_vec_t = std::remove_reference_t<decltype(unique_vendor_id_vec)>;
        struct Context {
            unique_vendor_id_vec_t* unique_vendor_id_vec;
            decltype(unique_vendor_id_vec->clbegin()) it;
            size_t idx;
            size_t idx_end;
            bool fetch_end;

            Context(unique_vendor_id_vec_t* unique_vendor_id_vec, size_t idx, size_t idx_end)
                : unique_vendor_id_vec(unique_vendor_id_vec),
                  idx(idx),
                  idx_end(idx_end),
                  fetch_end(false)
            {
            }

            void pin()
            {
                it.pin();
            }

            void unpin()
            {
                it.unpin();
            }

            bool fetched()
            {
                return it.at_local();
            }

            bool run(DereferenceScope& scope)
            {
                if (fetch_end) {
                    goto next;
                }
                while (idx < idx_end) {
                    it        = unique_vendor_id_vec->clbegin().nextn(idx);
                    fetch_end = true;
                    if (!it.async_fetch(scope)) {
                        return false;
                    }
                next:
                    std::cout << *it << ", ";
                    idx++;
                    fetch_end = false;
                }
                return true;
            }
        };
        RootDereferenceScope scope;
        const size_t vec_size = unique_vendor_id_vec.size();
        const size_t block    = (vec_size + CTX_COUNT - 1) / CTX_COUNT;
        SCOPED_INLINE_ASYNC_FOR(Context, size_t, i, 0, i < vec_size, i += block, scope)
        return Context(&unique_vendor_id_vec, i, std::min(i + block, vec_size));
        SCOPED_INLINE_ASYNC_FOR_END
    } else {
        ERROR("alg dont exist");
    }
    std::cout << '}' << std::endl;

    std::cout << std::endl;
}

template <Algorithm alg = DEFAULT_ALG>
void calculate_haversine_distance_column(StdDataFrame<uint64_t>& df)
{
    std::cout << "calculate_haversine_distance_column()" << std::endl;

    auto& pickup_longitude_vec  = df.get_column<double>("pickup_longitude");
    auto& pickup_latitude_vec   = df.get_column<double>("pickup_latitude");
    auto& dropoff_longitude_vec = df.get_column<double>("dropoff_longitude");
    auto& dropoff_latitude_vec  = df.get_column<double>("dropoff_latitude");
    assert(pickup_longitude_vec.size() == pickup_latitude_vec.size());
    assert(pickup_longitude_vec.size() == dropoff_longitude_vec.size());
    assert(pickup_longitude_vec.size() == dropoff_latitude_vec.size());
    FarLib::FarVector<double> haversine_distance_vec(pickup_longitude_vec.size());
    auto start = get_cycles();
    if constexpr (alg == DEFAULT) {
        for (uint64_t i = 0; i < pickup_longitude_vec.size(); i++) {
            *haversine_distance_vec[i] =
                (haversine(*pickup_latitude_vec[i], *pickup_longitude_vec[i],
                           *dropoff_latitude_vec[i], *dropoff_longitude_vec[i]));
        }
    } else if constexpr (alg == UTHREAD) {
        uthread::parallel_for_with_scope<1>(
            uthread::get_worker_count() * UTH_FACTOR, pickup_longitude_vec.size(),
            [&](size_t i, DereferenceScope& scope) {
                ON_MISS_BEGIN
                uthread::yield();
                ON_MISS_END
                *(haversine_distance_vec.at_mut(i, scope, __on_miss__)) =
                    haversine(*(pickup_latitude_vec.at(i, scope, __on_miss__)),
                              *(pickup_longitude_vec.at(i, scope, __on_miss__)),
                              *(dropoff_latitude_vec.at(i, scope, __on_miss__)),
                              *(dropoff_longitude_vec.at(i, scope, __on_miss__)));
            });
    } else if constexpr (alg == PREFETCH) {
        struct Scope : public RootDereferenceScope {
            decltype(pickup_latitude_vec.clbegin()) pickup_latitude_it;
            decltype(pickup_longitude_vec.clbegin()) pickup_longitude_it;
            decltype(dropoff_latitude_vec.clbegin()) dropoff_latitude_it;
            decltype(dropoff_longitude_vec.clbegin()) dropoff_longitude_it;
            decltype(haversine_distance_vec.lbegin()) haversine_it;

            void pin() const override
            {
                pickup_latitude_it.pin();
                pickup_longitude_it.pin();
                dropoff_latitude_it.pin();
                dropoff_longitude_it.pin();
                haversine_it.pin();
            }

            void unpin() const override
            {
                pickup_latitude_it.unpin();
                pickup_longitude_it.unpin();
                dropoff_latitude_it.unpin();
                dropoff_longitude_it.unpin();
                haversine_it.unpin();
            }

            void next()
            {
                pickup_latitude_it.next(*this);
                pickup_longitude_it.next(*this);
                dropoff_latitude_it.next(*this);
                dropoff_longitude_it.next(*this);
                haversine_it.next(*this);
            }
        } scope;
        scope.pickup_latitude_it   = pickup_latitude_vec.clbegin(scope);
        scope.pickup_longitude_it  = pickup_longitude_vec.clbegin(scope);
        scope.dropoff_latitude_it  = dropoff_latitude_vec.clbegin(scope);
        scope.dropoff_longitude_it = dropoff_longitude_vec.clbegin(scope);
        scope.haversine_it         = haversine_distance_vec.lbegin(scope);
        for (size_t i = 0; i < pickup_longitude_vec.size(); i++, scope.next()) {
            *(scope.haversine_it) =
                haversine(*(scope.pickup_latitude_it), *(scope.pickup_longitude_it),
                          *(scope.dropoff_latitude_it), *(scope.dropoff_longitude_it));
        }
    } else if constexpr (alg == PARAROUTINE) {
        using pickup_latitude_vec_t   = std::remove_reference_t<decltype(pickup_latitude_vec)>;
        using pickup_longitude_vec_t  = std::remove_reference_t<decltype(pickup_longitude_vec)>;
        using dropoff_latitude_vec_t  = std::remove_reference_t<decltype(dropoff_latitude_vec)>;
        using dropoff_longitude_vec_t = std::remove_reference_t<decltype(dropoff_longitude_vec)>;
        using haversine_vec_t         = std::remove_reference_t<decltype(haversine_distance_vec)>;

        struct Context {
            pickup_latitude_vec_t* pickup_latitude_vec;
            pickup_longitude_vec_t* pickup_longitude_vec;
            dropoff_latitude_vec_t* dropoff_latitude_vec;
            dropoff_longitude_vec_t* dropoff_longitude_vec;
            haversine_vec_t* haversine_vec;
            decltype(pickup_latitude_vec->clbegin()) pickup_latitude_it;
            decltype(pickup_longitude_vec->clbegin()) pickup_longitude_it;
            decltype(dropoff_latitude_vec->clbegin()) dropoff_latitude_it;
            decltype(dropoff_longitude_vec->clbegin()) dropoff_longitude_it;
            decltype(haversine_vec->lbegin()) haversine_it;
            size_t idx;
            size_t idx_end;
            double hav;
            enum Stage { INIT, LOCATION_FETCHING, HAVERSINE_FETCHING } stage;

            Context(pickup_latitude_vec_t* pickup_latitude_vec,
                    pickup_longitude_vec_t* pickup_longitude_vec,
                    dropoff_latitude_vec_t* dropoff_latitude_vec,
                    dropoff_longitude_vec_t* dropoff_longitude_vec, haversine_vec_t* haversine_vec,
                    size_t idx, size_t idx_end)
                : pickup_latitude_vec(pickup_latitude_vec),
                  pickup_longitude_vec(pickup_longitude_vec),
                  dropoff_latitude_vec(dropoff_latitude_vec),
                  dropoff_longitude_vec(dropoff_longitude_vec),
                  haversine_vec(haversine_vec),
                  idx(idx),
                  idx_end(idx_end),
                  stage(INIT)
            {
            }
            void pin()
            {
                pickup_latitude_it.pin();
                pickup_longitude_it.pin();
                dropoff_latitude_it.pin();
                dropoff_longitude_it.pin();
                haversine_it.pin();
            }
            void unpin()
            {
                pickup_latitude_it.unpin();
                pickup_longitude_it.unpin();
                dropoff_latitude_it.unpin();
                dropoff_longitude_it.unpin();
                haversine_it.unpin();
            }
            bool fetched()
            {
                switch (stage) {
                    case LOCATION_FETCHING:
                        return dropoff_longitude_it.at_local() && dropoff_latitude_it.at_local() &&
                               pickup_longitude_it.at_local() && pickup_latitude_it.at_local();
                    case HAVERSINE_FETCHING:
                        return haversine_it.at_local();
                    case INIT:
                        return true;
                    default:
                        ERROR("stage not exist");
                }
            }
            bool run(DereferenceScope& scope)
            {
                switch (stage) {
                    case INIT:
                        break;
                    case LOCATION_FETCHING:
                        goto location_fetched;
                    case HAVERSINE_FETCHING:
                        goto haversine_fetched;
                    default:
                        ERROR("stage not exist");
                }
                while (idx < idx_end) {
                    pickup_latitude_it = pickup_latitude_vec->clbegin().nextn(idx);
                    pickup_latitude_it.async_fetch(scope);
                    pickup_longitude_it = pickup_longitude_vec->clbegin().nextn(idx);
                    pickup_longitude_it.async_fetch(scope);
                    dropoff_latitude_it = dropoff_latitude_vec->clbegin().nextn(idx);
                    dropoff_latitude_it.async_fetch(scope);
                    dropoff_longitude_it = dropoff_longitude_vec->clbegin().nextn(idx);
                    dropoff_longitude_it.async_fetch(scope);
                    haversine_it = haversine_vec->lbegin().nextn(idx);
                    haversine_it.async_fetch(scope);
                    stage = LOCATION_FETCHING;
                    if (!fetched()) {
                        return false;
                    }
                location_fetched:
                    hav = haversine(*pickup_latitude_it, *pickup_longitude_it, *dropoff_latitude_it,
                                    *dropoff_longitude_it);
                    stage = HAVERSINE_FETCHING;
                    if (!fetched()) {
                        return false;
                    }
                haversine_fetched:
                    *haversine_it = hav;
                    idx++;
                    stage = INIT;
                }
                return true;
            }
        };
        RootDereferenceScope scope;
        const size_t block = (pickup_longitude_vec.size() + CTX_COUNT - 1) / CTX_COUNT;
        SCOPED_INLINE_ASYNC_FOR(Context, size_t, i, 0, i < pickup_longitude_vec.size(), i += block,
                                scope)
        return Context(&pickup_latitude_vec, &pickup_longitude_vec, &dropoff_latitude_vec,
                       &dropoff_longitude_vec, &haversine_distance_vec, i,
                       std::min(i + block, pickup_longitude_vec.size()));
        SCOPED_INLINE_ASYNC_FOR_END
    } else {
        ERROR("algorithm dont exist");
    }
    auto end = get_cycles();
    std::cout << "haversine cal " << end - start << std::endl;
    start = get_cycles();
    df.load_column<alg>("haversine_distance", std::move(haversine_distance_vec),
                        nan_policy::dont_pad_with_nans);
    end = get_cycles();
    std::cout << "haversine load column: " << end - start << std::endl;
    start            = get_cycles();
    auto sel_functor = [&](const uint64_t&, const double& dist) -> bool { return dist > 100; };
    auto sel_df =
        df.get_data_by_sel<alg, double, decltype(sel_functor), int, SimpleTime, double, char>(
            "haversine_distance", sel_functor);
    end = get_cycles();
    std::cout << "haversine sel df: " << end - start << std::endl;
    std::cout << "Number of rows that have haversine_distance > 100 KM = "
              << sel_df.get_index().size() << std::endl;

    std::cout << std::endl;
}

template <Algorithm alg = DEFAULT_ALG>
void analyze_trip_timestamp(StdDataFrame<uint64_t>& df)
{
    std::cout << "analyze_trip_timestamp()" << std::endl;
    auto start = get_cycles();
    MaxVisitor<SimpleTime> max_visitor;
    MinVisitor<SimpleTime> min_visitor;
    df.multi_visit<alg>(std::make_pair("tpep_pickup_datetime", &max_visitor),
                        std::make_pair("tpep_pickup_datetime", &min_visitor));
    std::cout << max_visitor.get_result() << std::endl;
    std::cout << min_visitor.get_result() << std::endl;
    auto end = get_cycles();
    std::cout << "ts visit: " << end - start << std::endl;
    start                 = get_cycles();
    auto& pickup_time_vec = df.get_column<SimpleTime>("tpep_pickup_datetime");
    FarLib::FarVector<char> pickup_hour_vec(pickup_time_vec.size());
    FarLib::FarVector<char> pickup_day_vec(pickup_time_vec.size());
    FarLib::FarVector<char> pickup_month_vec(pickup_time_vec.size());
    std::map<char, int> pickup_hour_map;
    std::map<char, int> pickup_day_map;
    std::map<char, int> pickup_month_map;

    if constexpr (alg == DEFAULT) {
        auto hour_it  = pickup_hour_vec.begin();
        auto day_it   = pickup_day_vec.begin();
        auto month_it = pickup_month_vec.begin();
        auto time_it  = pickup_time_vec.cbegin();

        for (uint64_t i = 0; i < pickup_time_vec.size();
             ++i, ++hour_it, ++day_it, ++month_it, ++time_it) {
            auto time = *time_it;
            pickup_hour_map[time.hour_]++;
            *hour_it = time.hour_;
            pickup_day_map[time.day_]++;
            *day_it = time.day_;
            pickup_month_map[time.month_]++;
            *month_it = time.month_;
        }
    } else {
        if constexpr (alg == UTHREAD) {
            uthread::parallel_for_with_scope<1>(
                uthread::get_worker_count() * UTH_FACTOR, pickup_time_vec.size(),
                [&](size_t i, DereferenceScope& scope) {
                    ON_MISS_BEGIN
                    uthread::yield();
                    ON_MISS_END
                    auto time = *(pickup_time_vec.at(i, scope, __on_miss__));
                    pickup_hour_map[time.hour_]++;
                    pickup_day_map[time.day_]++;
                    pickup_month_map[time.month_]++;
                    *(pickup_hour_vec.at_mut(i, scope, __on_miss__))  = time.hour_;
                    *(pickup_day_vec.at_mut(i, scope, __on_miss__))   = time.day_;
                    *(pickup_month_vec.at_mut(i, scope, __on_miss__)) = time.month_;
                });
        } else if constexpr (alg == PREFETCH) {
            struct Scope : public RootDereferenceScope {
                decltype(pickup_time_vec.clbegin()) pickup_time_it;
                decltype(pickup_hour_vec.lbegin()) pickup_hour_it;
                decltype(pickup_day_vec.lbegin()) pickup_day_it;
                decltype(pickup_month_vec.lbegin()) pickup_month_it;

                void pin() const override
                {
                    pickup_time_it.pin();
                    pickup_hour_it.pin();
                    pickup_day_it.pin();
                    pickup_month_it.pin();
                }

                void unpin() const override
                {
                    pickup_time_it.unpin();
                    pickup_hour_it.unpin();
                    pickup_day_it.unpin();
                    pickup_month_it.unpin();
                }

                void next()
                {
                    pickup_time_it.next(*this);
                    pickup_hour_it.next(*this);
                    pickup_day_it.next(*this);
                    pickup_month_it.next(*this);
                }
            } scope;
            scope.pickup_time_it  = pickup_time_vec.clbegin(scope);
            scope.pickup_hour_it  = pickup_hour_vec.lbegin(scope);
            scope.pickup_day_it   = pickup_day_vec.lbegin(scope);
            scope.pickup_month_it = pickup_month_vec.lbegin(scope);

            for (size_t i = 0; i < pickup_time_vec.size(); i++, scope.next()) {
                pickup_hour_map[scope.pickup_time_it->hour_]++;
                pickup_day_map[scope.pickup_time_it->day_]++;
                pickup_month_map[scope.pickup_time_it->month_]++;
                *(scope.pickup_hour_it)  = scope.pickup_time_it->hour_;
                *(scope.pickup_day_it)   = scope.pickup_time_it->day_;
                *(scope.pickup_month_it) = scope.pickup_time_it->month_;
            }
        } else if constexpr (alg == PARAROUTINE) {
            using time_vec_t  = std::remove_reference_t<decltype(pickup_time_vec)>;
            using hour_vec_t  = std::remove_reference_t<decltype(pickup_hour_vec)>;
            using day_vec_t   = std::remove_reference_t<decltype(pickup_day_vec)>;
            using month_vec_t = std::remove_reference_t<decltype(pickup_month_vec)>;
            using hour_map_t  = std::remove_reference_t<decltype(pickup_hour_map)>;
            using day_map_t   = std::remove_reference_t<decltype(pickup_day_map)>;
            using month_map_t = std::remove_reference_t<decltype(pickup_month_map)>;

            struct Context {
                time_vec_t* time_vec;
                hour_vec_t* hour_vec;
                day_vec_t* day_vec;
                month_vec_t* month_vec;
                hour_map_t* hour_map;
                day_map_t* day_map;
                month_map_t* month_map;
                decltype(time_vec->clbegin()) time_it;
                decltype(hour_vec->lbegin()) hour_it;
                decltype(day_vec->lbegin()) day_it;
                decltype(month_vec->lbegin()) month_it;
                size_t idx;
                size_t idx_end;
                enum Stage { TIME_FETCHING, TIME_FETCHED, ALL_FETCHED } stage;

                void pin()
                {
                    time_it.pin();
                    hour_it.pin();
                    day_it.pin();
                    month_it.pin();
                }
                void unpin()
                {
                    time_it.unpin();
                    hour_it.unpin();
                    day_it.unpin();
                    month_it.unpin();
                }

                bool fetched()
                {
                    switch (stage) {
                        case TIME_FETCHING:
                            return true;
                        case TIME_FETCHED:
                            return time_it.at_local();
                        case ALL_FETCHED:
                            return hour_it.at_local() && day_it.at_local() && month_it.at_local();
                        default:
                            ERROR("state error");
                    }
                }

                bool run(DereferenceScope& scope)
                {
                    switch (stage) {
                        case TIME_FETCHING:
                            break;
                        case TIME_FETCHED:
                            goto time_fetched;
                        case ALL_FETCHED:
                            goto all_fetched;
                        default:
                            ERROR("stage not exist");
                    }
                    while (idx < idx_end) {
                        time_it = time_vec->clbegin().nextn(idx);
                        time_it.async_fetch(scope);
                        hour_it = hour_vec->lbegin().nextn(idx);
                        hour_it.async_fetch(scope);
                        day_it = day_vec->lbegin().nextn(idx);
                        day_it.async_fetch(scope);
                        month_it = month_vec->lbegin().nextn(idx);
                        month_it.async_fetch(scope);
                        stage = TIME_FETCHED;
                        if (!fetched()) {
                            return false;
                        }
                    time_fetched:
                        (*hour_map)[time_it->hour_]++;
                        (*day_map)[time_it->day_]++;
                        (*month_map)[time_it->month_]++;
                        stage = ALL_FETCHED;
                        if (!fetched()) {
                            return false;
                        }
                    all_fetched:
                        *hour_it  = time_it->hour_;
                        *day_it   = time_it->day_;
                        *month_it = time_it->month_;
                        idx++;
                        stage = TIME_FETCHING;
                    }
                    return true;
                }
            };
            RootDereferenceScope scope;
            const size_t block = (pickup_time_vec.size() + CTX_COUNT - 1) / CTX_COUNT;
            SCOPED_INLINE_ASYNC_FOR(Context, size_t, i, 0, i < pickup_time_vec.size(), i += block,
                                    scope)
            return Context{
                .time_vec  = &pickup_time_vec,
                .hour_vec  = &pickup_hour_vec,
                .day_vec   = &pickup_day_vec,
                .month_vec = &pickup_month_vec,
                .hour_map  = &pickup_hour_map,
                .day_map   = &pickup_day_map,
                .month_map = &pickup_month_map,
                .idx       = i,
                .idx_end   = std::min(i + block, pickup_time_vec.size()),
                .stage     = Context::TIME_FETCHING,
            };
            SCOPED_INLINE_ASYNC_FOR_END
        } else {
            ERROR("algorithm dont exist");
        }
    }
    end = get_cycles();
    std::cout << "pickup time map fill: " << end - start << std::endl;
    start = get_cycles();
    df.load_column<alg>("pickup_hour", std::move(pickup_hour_vec), nan_policy::dont_pad_with_nans);
    df.load_column<alg>("pickup_day", std::move(pickup_day_vec), nan_policy::dont_pad_with_nans);
    df.load_column<alg>("pickup_month", std::move(pickup_month_vec),
                        nan_policy::dont_pad_with_nans);
    end = get_cycles();
    std::cout << "pickup time load column*3 " << end - start << std::endl;
    std::cout << "Print top 10 rows." << std::endl;
    start          = get_cycles();
    auto top_10_df = df.get_data_by_idx<int, SimpleTime, double, char>(
        Index2D<StdDataFrame<uint64_t>::IndexType>{0, 9});
    end = get_cycles();
    std::cout << "top10 df get: " << end - start << std::endl;
    start = get_cycles();
    top_10_df.write_with_values_only<std::ostream, int, SimpleTime, double, char>(std::cout, false,
                                                                                  io_format::json);
    std::cout << std::endl;
    end = get_cycles();
    std::cout << "top10 write: " << end - start << std::endl;

    for (auto& [hour, cnt] : pickup_hour_map) {
        std::cout << "pickup_hour = " << static_cast<int>(hour) << ", cnt = " << cnt << std::endl;
    }
    std::cout << std::endl;
    for (auto& [day, cnt] : pickup_day_map) {
        std::cout << "pickup_day = " << static_cast<int>(day) << ", cnt = " << cnt << std::endl;
    }
    std::cout << std::endl;
    for (auto& [month, cnt] : pickup_month_map) {
        std::cout << "pickup_month = " << static_cast<int>(month) << ", cnt = " << cnt << std::endl;
    }
    std::cout << std::endl;
}

template <typename T_Key, Algorithm alg = DEFAULT_ALG>
void analyze_trip_durations_of_timestamps(StdDataFrame<uint64_t>& df, const char* key_col_name)
{
    std::cout << "analyze_trip_durations_of_timestamps() on key = " << key_col_name << std::endl;

    StdDataFrame<uint64_t> df_key_duration;
    auto copy_index        = df.get_index();
    auto copy_key_col      = df.get_column<T_Key>(key_col_name);
    auto copy_key_duration = df.get_column<uint64_t>("duration");
    df_key_duration.load_data(std::move(copy_index),
                              std::make_pair(key_col_name, std::move(copy_key_col)),
                              std::make_pair("duration", std::move(copy_key_duration)));

    StdDataFrame<uint64_t> groupby_key =
        df_key_duration.groupby<alg, GroupbyMedian, T_Key, T_Key, uint64_t>(GroupbyMedian(),
                                                                            key_col_name);
    auto& key_vec      = groupby_key.get_column<T_Key>(key_col_name);
    auto& duration_vec = groupby_key.get_column<uint64_t>("duration");
    if constexpr (alg == DEFAULT) {
        for (uint64_t i = 0; i < key_vec.size(); i++) {
            std::cout << static_cast<int>(*key_vec[i]) << " " << *duration_vec[i] << std::endl;
        }
    } else if constexpr (alg == UTHREAD) {
        uthread::parallel_for_with_scope<1>(
            uthread::get_worker_count(), key_vec.size(), [&](size_t i, DereferenceScope& scope) {
                ON_MISS_BEGIN
                uthread::yield();
                ON_MISS_END
                auto key      = *(key_vec.at(i, scope, __on_miss__));
                auto duration = *(duration_vec.at(i, scope, __on_miss__));
                std::cout << static_cast<int>(key) << " " << duration << std::endl;
            });
    } else if constexpr (alg == PREFETCH) {
        struct Scope : public RootDereferenceScope {
            decltype(key_vec.clbegin()) key_it;
            decltype(duration_vec.clbegin()) duration_it;

            void pin() const override
            {
                key_it.pin();
                duration_it.pin();
            }

            void unpin() const override
            {
                key_it.unpin();
                duration_it.unpin();
            }

            void next()
            {
                key_it.next(*this);
                duration_it.next(*this);
            }
        } scope;
        scope.key_it      = key_vec.clbegin(scope);
        scope.duration_it = duration_vec.clbegin(scope);
        for (uint64_t i = 0; i < key_vec.size(); i++, scope.next()) {
            std::cout << static_cast<int>(*scope.key_it) << " " << *scope.duration_it << std::endl;
        }
    } else if constexpr (alg == PARAROUTINE) {
        using key_vec_t      = std::remove_reference_t<decltype(key_vec)>;
        using duration_vec_t = std::remove_reference_t<decltype(duration_vec)>;

        struct Context {
            key_vec_t* key_vec;
            duration_vec_t* duration_vec;
            decltype(key_vec->clbegin()) key_it;
            decltype(duration_vec->clbegin()) duration_it;
            size_t idx;
            size_t idx_end;
            bool fetch_end;

            void pin()
            {
                key_it.pin();
                duration_it.pin();
            }

            void unpin()
            {
                key_it.unpin();
                duration_it.unpin();
            }

            bool fetched()
            {
                return key_it.at_local() && duration_it.at_local();
            }

            bool run(DereferenceScope& scope)
            {
                if (fetch_end) {
                    goto next;
                }
                while (idx < idx_end) {
                    key_it = key_vec->clbegin().nextn(idx);
                    key_it.async_fetch(scope);
                    duration_it = duration_vec->clbegin().nextn(idx);
                    duration_it.async_fetch(scope);
                    fetch_end = true;
                    if (!fetched()) {
                        return false;
                    }
                next:
                    std::cout << static_cast<int>(*key_it) << " " << *duration_it << std::endl;
                    idx++;
                    fetch_end = false;
                }
                return true;
            }
        };
        RootDereferenceScope scope;
        const size_t block = (key_vec.size() + CTX_COUNT - 1) / CTX_COUNT;
        SCOPED_INLINE_ASYNC_FOR(Context, size_t, i, 0, i < key_vec.size(), i += block, scope)
        return Context{
            .key_vec      = &key_vec,
            .duration_vec = &duration_vec,
            .idx          = i,
            .idx_end      = std::min(i + block, key_vec.size()),
            .fetch_end    = false,
        };
        SCOPED_INLINE_ASYNC_FOR_END
    } else {
        ERROR("alg not exists");
    }
    std::cout << std::endl;
}

int main(int argc, const char* argv[])
{
    /* config setting */
    Configure config;
#ifdef STANDALONE
    config.server_addr = "127.0.0.1";
    config.server_port = "1234";
#ifdef SIMPLE_BENCH
    // ~74M
    config.server_buffer_size = 1024L * 1024 * 1024 * 2;
    config.client_buffer_size = 1024L * 1024 * 16;
#else
    // ~16G
    config.server_buffer_size = 1024L * 1024 * 1024 * 64;
    config.client_buffer_size = 1024L * 1024 * 1024 * 4;
#endif
    config.evict_batch_size = 64 * 1024;
#else
    if (argc != 2 && argc != 3) {
        std::cout << "usage: " << argv[0] << " <configure file> [client buffer size]" << std::endl;
        return -1;
    }
    config.from_file(argv[1]);
    if (argc == 3) {
        config.client_buffer_size = std::stoul(argv[2]);
    }
#endif

    /* client-server connection */
#ifdef STANDALONE
    Server server(config);
    std::thread server_thread([&server] { server.start(); });
    std::this_thread::sleep_for(1s);
#endif
    FarLib::runtime_init(config);
    srand(time(NULL));
    /* test */
    std::chrono::time_point<std::chrono::steady_clock> times[10];
    {
        FarLib::Cache::init_profile();
        perf_init();
        auto df  = load_data();
        times[0] = std::chrono::steady_clock::now();
        print_number_vendor_ids_and_unique(df);
        FarLib::Cache::start_profile();
        times[1] = std::chrono::steady_clock::now();
        print_passage_counts_by_vendor_id(df, 1);
        times[2] = std::chrono::steady_clock::now();
        FarLib::Cache::end_profile();
        // print_passage_counts_by_vendor_id(df, 2);
        // times[3] = std::chrono::steady_clock::now();
        // calculate_trip_duration(df);
        // times[4] = std::chrono::steady_clock::now();
        // calculate_distribution_store_and_fwd_flag(df);
        // times[5] = std::chrono::steady_clock::now();
        // calculate_haversine_distance_column(df);
        // times[6] = std::chrono::steady_clock::now();
        // analyze_trip_timestamp(df);
        // times[7] = std::chrono::steady_clock::now();
        // analyze_trip_durations_of_timestamps<char>(df, "pickup_day");
        // times[8] = std::chrono::steady_clock::now();
        // analyze_trip_durations_of_timestamps<char>(df, "pickup_month");
        // times[9] = std::chrono::steady_clock::now();

        for (uint32_t i = 1; i < std::size(times); i++) {
            std::cout << "Step " << i << ": "
                      << std::chrono::duration_cast<std::chrono::microseconds>(times[i] -
                                                                               times[i - 1])
                             .count()
                      << " us" << std::endl;
        }
        std::cout
            << "Total: "
            << std::chrono::duration_cast<std::chrono::microseconds>(times[9] - times[0]).count()
            << " us" << std::endl;
    }
    /* destroy runtime */
    std::cout << "handle time: " << (static_cast<double>(FarLib::Cache::handle_time) / 2.8 / 1000)
              << "us" << std::endl;
    std::cout << "evict time: " << (static_cast<double>(FarLib::Cache::evict_time) / 2.8 / 1000)
              << "us" << std::endl;

    FarLib::runtime_destroy();
#ifdef STANDALONE
    server_thread.join();
#endif
    return 0;
}
