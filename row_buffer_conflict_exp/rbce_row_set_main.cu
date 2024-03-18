#include "rbce_n_conflict_exp.cuh"
#include <iostream>
#include <vector>

int main(int argc, char *argv[])
{
  /* Argument EXP_RANGE, EXP_IT, STEP */
  rbce::N_Conflict nc_test(2, 0, std::stoull(argv[1]), 32);

  /* Argument Threshold */
  uint64_t threshold = std::stoull(argv[2]);

  /* Offset to an address in Target Bank */
  uint64_t offset_to_bank = std::stoull(argv[3]);

  /* Maximum rows we want */
  uint64_t max_row = std::stoull(argv[4]);

  const uint8_t **addr = new const uint8_t *[2];
  const uint8_t *addr_start = nc_test.get_addr_layout();
  std::vector<const uint8_t *> conf_vec{addr_start + offset_to_bank};

  /* Initialize address pairs */
  addr[0] = addr_start;
  addr[1] = addr_start + 1;
  nc_test.repeat_n_addr_exp(addr, NULL);

  std::string buf;
  std::ifstream conf_set_file(argv[5]);

  while (conf_set_file.peek() != EOF && max_row != conf_vec.size())
  {
    while (std::getline(conf_set_file, buf))
    {
      uint64_t conf_step = std::stoull(buf.c_str());

      /* Address to be tested for whether it should be in conf_set */
      addr[0] = addr_start + conf_step * 8;

      bool all_conflict = true;
      for (auto it = conf_vec.rbegin();
           it != conf_vec.rend() && all_conflict && it != conf_vec.rbegin() +
           5; it++)
      {
        /* Addresses already in conf_set */
        addr[1] = *it;
        uint64_t time = nc_test.repeat_n_addr_exp(addr, NULL);
        all_conflict &= (time > threshold);
      }

      if (all_conflict)
      {
        conf_vec.push_back(addr[0]);
        break;
      }
    }
  }

  std::ofstream row_set_file(argv[6]);
  const uint8_t *initial = addr_start + offset_to_bank;
  
  row_set_file << (void *)(initial) << '\t' << offset_to_bank << '\n';
  for (auto it = conf_vec.begin() + 1; it != conf_vec.end(); it++)
  {
    row_set_file << (const void *)(*it) << '\t' << *it - initial << '\n';
    initial = *it;
  }
  delete[] addr;
  conf_set_file.close();
  return 0;
}