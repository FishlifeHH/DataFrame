// Hossein Moein
// September 12, 2017
// Copyright (C) 2018-2019 Hossein Moein
// Distributed under the BSD Software License (see file License)

#include "DataFrame.h"
#include <tuple>

// ----------------------------------------------------------------------------

namespace hmdf
{

// ----------------------------------------------------------------------------

template<typename TS, typename HETERO>
template<typename RHS_T, typename ... types>
StdDataFrame<TS> DataFrame<TS, HETERO>::
merge_by_index (const RHS_T &rhs, merge_policy mp) const  {

    static_assert(std::is_base_of<StdDataFrame<TS>, RHS_T>::value or
                      std::is_base_of<DataFrameView<TS>, RHS_T>::value,
                  "The rhs argument to merge_by_index() can only be "
                  "StdDataFrame<TS> or DataFrameView<TS>");

    switch(mp)  {
        case merge_policy::inner_join:
            return (index_inner_join_
                        <decltype(*this), decltype(rhs), types ...>
                            (*this, rhs));
            break;
        case merge_policy::left_join:
            return (index_left_join_
                        <decltype(*this), decltype(rhs), types ...>
                            (*this, rhs));
            break;
        case merge_policy::right_join:
            return (index_right_join_
                        <decltype(*this), decltype(rhs), types ...>
                            (*this, rhs));
            break;
        case merge_policy::left_right_join:
        default:
            return (index_left_right_join_
                        <decltype(*this), decltype(rhs), types ...>
                            (*this, rhs));
            break;
    }
}

// ----------------------------------------------------------------------------

template<typename TS, typename HETERO>
template<typename RHS_T, typename T, typename ... types>
StdDataFrame<TS> DataFrame<TS, HETERO>::
merge_by_column (const RHS_T &rhs,
                 const char *col_name,
                 merge_policy mp) const {

    static_assert(std::is_base_of<StdDataFrame<TS>, RHS_T>::value or
                      std::is_base_of<DataFrameView<TS>, RHS_T>::value,
                  "The rhs argument to merge_by_column() can only be "
                  "StdDataFrame<TS> or DataFrameView<TS>");

    switch(mp)  {
        case merge_policy::inner_join:
            return (column_inner_join_
                        <decltype(*this), decltype(rhs), T, types ...>
                            (col_name, *this, rhs));
            break;
        case merge_policy::left_join:
            return (column_left_join_
                        <decltype(*this), decltype(rhs), T, types ...>
                            (col_name, *this, rhs));
            break;
        case merge_policy::right_join:
            return (column_right_join_
                        <decltype(*this), decltype(rhs), T, types ...>
                            (col_name, *this, rhs));
            break;
        case merge_policy::left_right_join:
        default:
            return (column_left_right_join_
                        <decltype(*this), decltype(rhs), T, types ...>
                            (col_name, *this, rhs));
            break;
    }
}

// ----------------------------------------------------------------------------

template<typename TS, typename HETERO>
template<typename LHS_T, typename RHS_T, typename ... types>
StdDataFrame<TS> DataFrame<TS, HETERO>::
index_inner_join_(const LHS_T &lhs, const RHS_T &rhs)  {

    size_type       lhs_current = 0;
    const size_type lhs_end = lhs.indices_.size();
    size_type       rhs_current = 0;
    const size_type rhs_end = rhs.indices_.size();

    std::vector<std::tuple<size_type, size_type>>   merged_index_idx;

    merged_index_idx.reserve(std::min(lhs_end, rhs_end));
    while (lhs_current != lhs_end && rhs_current != rhs_end) {
        if (lhs.indices_[lhs_current] < rhs.indices_[rhs_current])
            lhs_current += 1;
        else  {
            if (lhs.indices_[lhs_current] == rhs.indices_[rhs_current])
                merged_index_idx.emplace_back(lhs_current++, rhs_current);
            rhs_current += 1;
        }
    }

    StdDataFrame<TS>    result;
    std::vector<TS>     result_index;

    result_index.reserve(merged_index_idx.size());
    for (const auto &citer : merged_index_idx)
        result_index.push_back(std::get<0>(citer));
    result.load_index(std::move(result_index));

    for (auto &iter : lhs.data_tb_)
        if (rhs.data_tb_.find(iter.first) != rhs.data_tb_.end())  {
            index_join_functor_<types ...>  functor (iter.first.c_str(),
                                                     rhs,
                                                     merged_index_idx,
                                                     result);

            lhs.data_[iter.second].change(functor);

        }

    return(result);
}

// ----------------------------------------------------------------------------

template<typename TS, typename HETERO>
template<typename LHS_T, typename RHS_T, typename ... types>
StdDataFrame<TS> DataFrame<TS, HETERO>::
index_left_join_(const LHS_T &lhs, const RHS_T &rhs)  {

    StdDataFrame<TS>    result;

    return(result);
}

// ----------------------------------------------------------------------------

template<typename TS, typename HETERO>
template<typename LHS_T, typename RHS_T, typename ... types>
StdDataFrame<TS> DataFrame<TS, HETERO>::
index_right_join_(const LHS_T &lhs, const RHS_T &rhs)  {

    StdDataFrame<TS>    result;

    return(result);
}

// ----------------------------------------------------------------------------

template<typename TS, typename HETERO>
template<typename LHS_T, typename RHS_T, typename ... types>
StdDataFrame<TS> DataFrame<TS, HETERO>::
index_left_right_join_(const LHS_T &lhs, const RHS_T &rhs)  {

    StdDataFrame<TS>    result;

    return(result);
}

// ----------------------------------------------------------------------------

template<typename TS, typename HETERO>
template<typename LHS_T, typename RHS_T, typename COL_T, typename ... types>
StdDataFrame<TS> DataFrame<TS, HETERO>::
column_inner_join_(const char *col_name, const LHS_T &lhs, const RHS_T &rhs)  {

    StdDataFrame<TS>    result;

    return(result);
}

// ----------------------------------------------------------------------------

template<typename TS, typename HETERO>
template<typename LHS_T, typename RHS_T, typename COL_T, typename ... types>
StdDataFrame<TS> DataFrame<TS, HETERO>::
column_left_join_(const char *col_name, const LHS_T &lhs, const RHS_T &rhs)  {

    StdDataFrame<TS>    result;

    return(result);
}

// ----------------------------------------------------------------------------

template<typename TS, typename HETERO>
template<typename LHS_T, typename RHS_T, typename COL_T, typename ... types>
StdDataFrame<TS> DataFrame<TS, HETERO>::
column_right_join_(const char *col_name, const LHS_T &lhs, const RHS_T &rhs)  {

    StdDataFrame<TS>    result;

    return(result);
}

// ----------------------------------------------------------------------------

template<typename TS, typename HETERO>
template<typename LHS_T, typename RHS_T, typename COL_T, typename ... types>
StdDataFrame<TS> DataFrame<TS, HETERO>::
column_left_right_join_(const char *col_name,
                        const LHS_T &lhs,
                        const RHS_T &rhs)  {

    StdDataFrame<TS>    result;

    return(result);
}

} // namespace hmdf

// ----------------------------------------------------------------------------

// Local Variables:
// mode:C++
// tab-width:4
// c-basic-offset:4
// End:
