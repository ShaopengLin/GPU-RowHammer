#include "rbce_main.cuh"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
namespace rbce
{

void run_two_conflict_exp(N_Conflict &nc_test)
{
  std::ofstream myfile;
  myfile.open("8MB.txt");
  const uint64_t **addr = new const uint64_t *[2];
  addr[0] = nc_test.get_addr_layout();
  addr[1] = nc_test.get_addr_layout() + 1;
  nc_test.repeat_n_addr_exp(addr, NULL);

  uint64_t step = 0;
  for (; step < nc_test.get_exp_range() / 8; step += nc_test.STEP_SIZE)
  {
    addr[1] = nc_test.get_addr_layout() + step;
    nc_test.repeat_n_addr_exp(addr, &myfile);
  }
  delete[] addr;
  myfile.close();
}

std::vector<const uint64_t *> find_n_rows(N_Conflict &nc_test, uint64_t num_row,
                                          uint64_t threshold)
{
  const uint64_t *addr_start = nc_test.get_addr_layout();
  std::vector<const uint64_t *> conf_vec{addr_start};
  const uint64_t **addr = new const uint64_t *[2];

  uint64_t step = 0;
  while (conf_vec.size() < num_row)
  {
    for (; step < nc_test.LAYOUT_SIZE / 8; step += nc_test.STEP_SIZE)
    {
      addr[0] = addr_start + step;
      if (std::accumulate(conf_vec.rbegin(), conf_vec.rend(), true,
                          [&nc_test, &conf_vec, addr, threshold](
                              bool all_conflict, const uint64_t *conf_addr)
                          {
                            addr[1] = conf_addr;
                            uint64_t time =
                                nc_test.repeat_n_addr_exp(addr, NULL);
                            return all_conflict && (time > threshold);
                          }))
      {
        conf_vec.push_back(addr[0]);
        break;
      }
    }
  }

  const uint64_t *initial = conf_vec[0];
  for (auto it = conf_vec.begin() + 1; it != conf_vec.end(); it++)
  {
    std::cout << " Conflict: " << *it << ", Offset: " << (*it - initial) / 256
              << '\n';
    initial = *it;
  }

  return conf_vec;
}

} // namespace rbce

int main(int argc, char *argv[])
{
  if (argc < 2)
    return -1;

  rbce::N_Conflict nc_test(argc, argv);
  if (argv[1] == rbce::TWO_CONFLICT)
    run_two_conflict_exp(nc_test);
  else if (argv[1] == rbce::FIND_ROW && argc >= 7)
  {
    find_n_rows(nc_test, std::stoull(argv[5]), std::stoull(argv[6]));
  }
  else
  {
    run_two_conflict_exp(nc_test);
    find_n_rows(nc_test, std::stoull(argv[5]), std::stoull(argv[6]));
  }

  return 0;
}
