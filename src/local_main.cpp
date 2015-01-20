/*!
 * \file local_main.cpp
 * \brief local main file that redirect directly to PSMain
 * \author Tianqi Chen
 */
int PSMain(int argc, char *argv[]);

int main(int argc, char *argv[]) {
  return PSMain(argc, argv);
}
