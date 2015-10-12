#include "HandDetection.h"

#include <numeric>
#include <regex>

#include <regex>
#include <cstring>
#include <sys/types.h>
#include <dirent.h>

using std::string;
using std::vector;
using std::accumulate;
using std::transform;

vector<string> findFilesLike(std::regex needle, string baseDir = ".") {
    const char *dir = baseDir.c_str();
    vector<string> results = {};
    DIR *dirp = opendir(dir);
    struct dirent *dp;
    while ((dp = readdir(dirp)) != NULL) {
        string entry = string(dp->d_name);
        if (std::regex_match(entry, needle))
            results.push_back(entry);
    }
    closedir(dirp);
    return results;
}

vector<string> findTestFiles() {
    string dir = "/Users/robert/Lively/LivelyKernel2/opencv-test/test-images/";
    std::regex needle("^hand-test-[0-9]+\\.[a-z]+$");
    // std::regex needle("^hand-test-2\\.[a-z]+$");
    auto found = findFilesLike(needle, dir);
    transform(found.begin(), found.end(), found.begin(),
        [dir](string ea){ return string(dir + ea); });
    return found;
}