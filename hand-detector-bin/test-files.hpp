#ifndef TEST_FILES_H_INCLUDE
#define TEST_FILES_H_INCLUDE

#include <stdio.h>
#include <regex>

std::vector<std::string> findTestFiles(std::regex = std::regex("^hand-test-[0-9]+\\.[a-z]+$"));

#endif  // TEST_FILES_H_INCLUDE

