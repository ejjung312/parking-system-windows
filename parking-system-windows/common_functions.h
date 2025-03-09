#pragma once

#include <windows.h>
#include <string>

#include "objectbbox.h"

std::string GetResourcePath(const std::string& filename);

float calculateIoU(const ObjectBBox& box1, const ObjectBBox& box2);