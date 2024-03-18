#include "rbce_n_conflict_exp.cuh"
#include <iostream>
#include <unordered_map>
#include <vector>

/* Idea:
     - Iterate through each address
     - for each address, find an address that conflict (same bank)
     - Check if any of the addresses in the bank set conflict with that address.
        If no, then it is a new bank and add it to bank set
     - (Also keep each address found and seen in a seen_set. If has a hit,
        remove it to save space.)
     - Not sure how to know how many banks there are yet but can just use a
        max for it.
*/
int main(int argc, char *argv[])
{
  /* Argument EXP_RANGE, EXP_IT, STEP */
  rbce::N_Conflict nc_test(2, std::stoull(argv[1]), std::stoull(argv[2]),
                           std::stoull(argv[3]));

  /* Argument Threshold */
  uint64_t threshold = std::stoull(argv[4]);

  /* Maximum banks we want */
  uint64_t max_bank = std::stoull(argv[5]);

  const uint8_t **addr = new const uint8_t *[2];
  const uint8_t *addr_start = nc_test.get_addr_layout();

  std::vector<const uint8_t *> bank_vec{addr_start};
  std::unordered_map<const uint8_t *, bool> seen_set;

  /* Initialize address pairs */
  addr[0] = addr_start;
  addr[1] = addr_start + 1;
  nc_test.repeat_n_addr_exp(addr, NULL);

  auto is_conf = [&](const uint8_t *current)
  {
    addr[0] = current;

    for (auto it = bank_vec.rbegin(); it != bank_vec.rend(); it++)
    {
      addr[1] = *it;
      if (threshold < nc_test.repeat_n_addr_exp(addr, NULL))
        return true;
    }
    return false;
  };

  auto find_conf = [&](uint64_t step, const uint8_t *current)
  {
    uint64_t tmp_step = step + nc_test.STEP_SIZE;
    addr[0] = current;
    for (; tmp_step < nc_test.get_exp_range(); tmp_step += nc_test.STEP_SIZE)
    {
      addr[1] = addr_start + tmp_step;
      auto out = nc_test.repeat_n_addr_exp(addr, NULL);
      if (threshold < out)
        return addr[1];
      std::cout << tmp_step << '\t' << out << '\n';
    }
    return (const uint8_t *)0;
  };

  uint64_t step = nc_test.STEP_SIZE;
  for (; step < nc_test.get_exp_range() && max_bank != bank_vec.size();
       step += nc_test.STEP_SIZE)
  {
    const uint8_t *current_addr = addr_start + step;

    /* We know this address must be in a found bank */
    if (seen_set.find(current_addr) != seen_set.end())
    {
      seen_set.erase(current_addr);
      continue;
    }

    std::cout << "S:" << step << '\n';
    /* New address, if conflict with bank_vec, it is in a found bank */
    if (is_conf(current_addr))
      continue;
    std::cout << "OK" << '\n';
    /* Find address in same bank as New address (In case New address is same
       row as one in bank_vec). New address is definitly different row */
    const uint8_t *conf_addr = find_conf(step, current_addr);
    std::cout << "Conf Addr:" << conf_addr - addr_start << '\n';
    seen_set[conf_addr] = false;
    if (is_conf(conf_addr))
      continue;

    bank_vec.push_back(current_addr);
    std::cout << step << '\n';
  }

  std::ofstream banks_file(argv[6]);
  const uint8_t *prev = addr_start;
  for (const auto addr : bank_vec)
  {
    banks_file << addr - prev << '\n';
    prev = addr;
  }

  banks_file.close();

  return 0;
}