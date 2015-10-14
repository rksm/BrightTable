#ifndef testFILES_H_INCLUDE
#define testFILES_H_INCLUDE

#include <stdio.h>
#include <regex>

std::vector<std::string> findTestFiles(std::regex = std::regex("^hand-test-[0-9]+\\.[a-z]+$"));

#endif  // testFILES_H_INCLUDE

