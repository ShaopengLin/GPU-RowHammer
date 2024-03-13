#include "rbce_util.cuh"
#include <fstream>
#include <stdint.h>
#ifndef GPU_ROWHAMMER_RBCE_N_CONFLICT_EXP_H
#define GPU_ROWHAMMER_RBCE_N_CONFLICT_EXP_H

namespace rbce
{

class N_Conflict
{
private:
  uint64_t *TIME_ARR_DEVICE;
  uint64_t *TIME_ARR_HOST;
  uint8_t *ADDR_LAYOUT;
  uint8_t **ADDR_LST_BUF;
  uint64_t N = 2;               /* Arg3 */
  uint64_t EXP_RANGE = 8388608; /* Arg4 */
  uint64_t EXP_IT = 10;         /* Arg5 */

  uint64_t CLOCK_RATE;

public:
  const uint64_t LAYOUT_SIZE = 16106127360;
  const uint64_t STEP_SIZE = 8; /* In Bytes */

  N_Conflict(int argc, char *argv[]);
  N_Conflict(uint64_t N, uint64_t EXP_RANGE, uint64_t EXP_IT,
             uint64_t STEP_SIZE = 4);
  ~N_Conflict();

  const uint8_t *get_addr_layout() { return ADDR_LAYOUT; };
  uint64_t get_exp_range() { return EXP_RANGE; };
  void set_exp_range(uint64_t EXP_RANGE) { this->EXP_RANGE = EXP_RANGE; };
  /**
   * @brief Runs the device code to access a different address depending on
   * tid.x. The code is written with the assumption that you are running it
   * within 1 single block with <= 32 threads.
   *
   * @param addr_arr array of addresses to access.
   * @param time_arr storage of time spent.
   */

  uint64_t repeat_n_addr_exp(const uint8_t **addr_arr, std::ofstream *file);
};

} // namespace rbce
#endif /* GPU_ROWHAMMER_RBCE_N_CONFLICT_EXP_H */