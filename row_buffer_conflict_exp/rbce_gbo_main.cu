#include "rbce_n_conflict_exp.cuh"
#include <iostream>

int main(int argc, char *argv[])
{
  /* Argument EXP_RANGE, EXP_IT, STEP */
  rbce::N_Conflict nc_test(2, std::stoull(argv[1]), std::stoull(argv[2]),
                           std::stoull(argv[3]));

  /* Argument Threshold */
  uint64_t threshold = std::stoull(argv[4]);

  /* Offset to an address in Target Bank */
  uint64_t offset_to_bank = std::stoull(argv[5]);

  std::ofstream offset_file;
  offset_file.open(argv[6]); /* Argument File name */
  const uint8_t **addr = new const uint8_t *[2];
  const uint8_t *addr_start = nc_test.get_addr_layout();

  /* Initialize address pairs */
  addr[0] = addr_start + offset_to_bank;
  addr[1] = addr_start + 1;
  nc_test.repeat_n_addr_exp(addr, NULL);

  uint64_t step = 0;
  for (; step < nc_test.get_exp_range(); step += nc_test.STEP_SIZE)
  {
    addr[1] = addr_start + step;
    if (threshold < nc_test.repeat_n_addr_exp(addr, NULL))
      offset_file << step << '\n';
  }

  offset_file.close();

  return 0;
}