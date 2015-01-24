/*!
 * \file local_main.cpp
 * \brief local main file that redirect directly to PSMain
 * \author Tianqi Chen
 */

int PS::WorkerNodeMain(int argc, char *argv[]);

int main(int argc, char *argv[]) {
  return PS::WorkerNodeMain(argc, argv);
}
