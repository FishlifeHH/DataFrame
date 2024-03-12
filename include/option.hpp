#pragma once
#include <cstddef>
enum Algorithm { DEFAULT, UTHREAD, PARAROUTINE, PREFETCH };
static constexpr size_t UTH_FACTOR     = 1;
static constexpr Algorithm DEFAULT_ALG = PREFETCH;