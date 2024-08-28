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

StdDataFrame<uint64_t> load_data()
{
#ifdef SIMPLE_BENCH
    // const char* file_path =
    // "/home/huanghong/mem_parallel/motivation/FarLib/build/very_simple.csv";
    const char* file_path = "/mnt/ssd/huanghong/data/simple.csv";

#else
    const char* file_path = "/mnt/ssd/huanghong/data/all.csv";
#endif
    return read_csv<-1, int, SimpleTime, SimpleTime, int, double, double, double, int, char, double,
                    double, int, double, double, double, double, double, double, double>(
        file_path, "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
        "trip_distance", "pickup_longitude", "pickup_latitude", "RatecodeID", "store_and_fwd_flag",
        "dropoff_longitude", "dropoff_latitude", "payment_type", "fare_amount", "extra", "mta_tax",
        "tip_amount", "tolls_amount", "improvement_surcharge", "total_amount");
}

template <Algorithm alg = DEFAULT_ALG>
void print_hours_and_unique(StdDataFrame<uint64_t>& df, size_t uthread_cnt)
{
    std::cout << "print_hours_and_unique()" << std::endl;
    std::cout << "Number of hours in the train dataset: "
              << df.get_column<SimpleTime>("tpep_pickup_datetime").size() << std::endl;
    size_t siz = df.get_col_unique_values<alg, SimpleTime>(
                       "tpep_pickup_datetime",
                       [](const SimpleTime& st) {
                           constexpr uint64_t day_per_month[12] = {31, 28, 31, 30, 31, 30,
                                                                   31, 31, 30, 31, 30, 31};
                           size_t days                          = 0;
                           for (size_t i = 0; i < st.month_ - 1; i++) {
                               days += day_per_month[i];
                           }
                           days += st.day_;
                           return days * 24 + st.hour_;
                       },
                       uthread_cnt)
                     .size();
    std::cout << "Number of unique hours in the train dataset:" << siz << std::endl;
    std::cout << std::endl;
}

int main(int argc, const char* argv[])
{
    /* config setting */
    Configure config;
    size_t uthread_cnt;
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
    if (argc != 2 && argc != 4) {
        std::cout << "usage: " << argv[0] << " <configure file> [<core num> <uthread num>]"
                  << std::endl;
        return -1;
    }
    config.from_file(argv[1]);
    if (argc == 4) {
        config.max_thread_cnt = std::stoul(argv[2]);
        uthread_cnt           = std::stoul(argv[3]);
    } else {
        uthread_cnt = config.max_thread_cnt * UTH_FACTOR;
    }
#endif

    /* client-server connection */
#ifdef STANDALONE
    Server server(config);
    std::thread server_thread([&server] { server.start(); });
    std::this_thread::sleep_for(1s);
#endif

    FarLib::runtime_init(config);
    /* test */
    std::chrono::time_point<std::chrono::steady_clock> times[10];

    {
        auto df = load_data();
        print_hours_and_unique(df, uthread_cnt);
    }
    /* destroy runtime */
    FarLib::runtime_destroy();
#ifdef STANDALONE
    server_thread.join();
#endif
    return 0;
}
