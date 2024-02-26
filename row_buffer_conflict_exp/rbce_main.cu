#include "rbce_main.cuh"
#include <algorithm>
#include <cstdlib>
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

void find_same_bank_addr(N_Conflict &nc_test, uint64_t threshold)
{
  std::ofstream myfile;
  myfile.open(rbce::BANK_SET_FILE_NAME);
  const uint64_t **addr = new const uint64_t *[2];
  const uint64_t *addr_start = nc_test.get_addr_layout();
  addr[0] = addr_start;
  addr[1] = addr_start + 1;
  nc_test.repeat_n_addr_exp(addr, NULL);

  uint64_t step = 0;
  for (; step < nc_test.get_exp_range() / 8; step += nc_test.STEP_SIZE)
  {
    addr[1] = addr_start + step;
    if (threshold < nc_test.repeat_n_addr_exp(addr, NULL))
      myfile << step << '\n';
  }

  myfile.close();

  delete[] addr;
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
      bool all_conflict = true;
      for (auto it = conf_vec.rbegin();
           it != conf_vec.rend() && all_conflict && it - conf_vec.rbegin() <= 5;
           it++)
      {
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

  const uint64_t *initial = conf_vec[0];
  for (auto it = conf_vec.begin() + 1; it != conf_vec.end(); it++)
  {
    std::cout << " Conflict: " << *it << ", Offset: " << (*it - initial) / 256
              << '\n';
    initial = *it;
  }
  delete[] addr;
  return conf_vec;
}

std::vector<const uint64_t *>
find_n_rows_file(N_Conflict &nc_test, uint64_t num_row, uint64_t threshold)
{
  const uint64_t *addr_start = nc_test.get_addr_layout();
  std::vector<const uint64_t *> conf_vec{addr_start};
  const uint64_t **addr = new const uint64_t *[2];
  addr[0] = nc_test.get_addr_layout();
  addr[1] = nc_test.get_addr_layout() + 1;
  nc_test.repeat_n_addr_exp(addr, NULL);

  std::string buf;
  std::ifstream bank_set_file(rbce::BANK_SET_FILE_NAME);

  while (conf_vec.size() < num_row)
  {
    while (std::getline(bank_set_file, buf))
    {
      uint64_t conf_step = std::stoull(buf.c_str());
      addr[0] = addr_start + conf_step;

      bool all_conflict = true;
      for (auto it = conf_vec.rbegin();
           it != conf_vec.rend() && all_conflict && it != conf_vec.rbegin() + 5;
           it++)
      {
        addr[1] = *it;
        uint64_t time = nc_test.repeat_n_addr_exp(addr, NULL);
        all_conflict &= (time > threshold);
        std::cout << all_conflict << '\n';
      }

      if (all_conflict)
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

  delete[] addr;
  bank_set_file.close();
  return conf_vec;
}

bool test_random_row_conflict(N_Conflict &nc_test,
                              std::vector<const uint64_t *> conf_vec,
                              uint64_t num_row, uint64_t threshold)
{
  srand(time(0));
  uint64_t random_row_idx = rand() % num_row;

  const uint64_t *addr_start = nc_test.get_addr_layout();
  const uint64_t **addr = new const uint64_t *[2];
  addr[0] = conf_vec[random_row_idx];

  return std::accumulate(
      conf_vec.begin(), conf_vec.end(), true,
      [&nc_test, addr, threshold](bool all_conflict, const uint64_t *conf_addr)
      {
        if (conf_addr == addr[0])
          return all_conflict;
        std::cout << conf_addr << '\n';
        addr[1] = conf_addr;
        uint64_t time = nc_test.repeat_n_addr_exp(addr, NULL);
        std::cout << time << '\n';
        return all_conflict && (time > threshold);
      });
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
  else if (argv[1] == rbce::FIND_SAME_BANK_ADDR && argc >= 7)
  {
    find_same_bank_addr(nc_test, std::stoull(argv[6]));
  }
  else if (argv[1] == rbce::FIND_ROW_FILE && argc >= 7)
  {
    std::cout << test_random_row_conflict(
                     nc_test,
                     find_n_rows_file(nc_test, std::stoull(argv[5]),
                                      std::stoull(argv[6])),
                     std::stoull(argv[5]), std::stoull(argv[6]))
              << '\n';
  }
  else
  {
    find_same_bank_addr(nc_test, std::stoull(argv[6]));
    find_n_rows(nc_test, std::stoull(argv[5]), std::stoull(argv[6]));
  }

  return 0;
}
