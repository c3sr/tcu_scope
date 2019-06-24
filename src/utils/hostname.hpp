#pragma once

#include <limits.h>
#include <stdio.h>
#include <string>
#include <unistd.h>

static std::string hostname() {
  char hostname[1024];
  gethostname(hostname, sizeof(hostname));
  hostname[sizeof(hostname) - 1] = 0;
  return std::string(hostname);
}
