/*!
 * \file local_main.cpp
 * \brief local main file that redirect directly to PSMain
 * \author Tianqi Chen
 */

int WorkerNodeMain(int argc, char *argv[]);

int main(int argc, char *argv[]) {
  return WorkerNodeMain(argc, argv);
}
