// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "CAparsefiles.hpp"

#include "mpi.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <regex>

// Functions that are used to simplify the parsing of input files, either by ExaCA or related utilities

//*****************************************************************************/
// Remove whitespace from "line", optional argument to take only portion of the line after position "pos"
std::string removeWhitespace(std::string line, int pos) {

    std::string val = line.substr(pos + 1, std::string::npos);
    std::regex r("\\s+");
    val = std::regex_replace(val, r, "");
    return val;
}

// Check if a string is Y (true) or N (false)
bool getInputBool(std::string val_input) {
    std::string val = removeWhitespace(val_input);
    if (val == "N") {
        return false;
    }
    else if (val == "Y") {
        return true;
    }
    else {
        std::string error = "Input \"" + val + "\" must be \"Y\" or \"N\".";
        throw std::runtime_error(error);
    }
}

// Convert string "val_input" to base 10 integer
int getInputInt(std::string val_input) {
    int IntFromString = stoi(val_input, nullptr, 10);
    return IntFromString;
}

// Convert string "val_input" to float value multiplied by 10^(factor)
float getInputFloat(std::string val_input, int factor) {
    float FloatFromString = atof(val_input.c_str()) * pow(10, factor);
    return FloatFromString;
}

// Convert string "val_input" to double value multiplied by 10^(factor)
double getInputDouble(std::string val_input, int factor) {
    double DoubleFromString = std::stod(val_input.c_str()) * pow(10, factor);
    return DoubleFromString;
}

// Given a string ("line"), parse at "separator" (commas used by default)
// Modifies "parsed_line" to hold the separated values
// expected_num_values may be larger than parsed_line_size, if only a portion of the line is being parsed
void splitString(const std::string line, std::vector<std::string> &parsed_line, std::size_t expected_num_values,
                 char separator) {
    // Make sure the right number of values are present on the line - one more than the number of separators
    std::size_t actual_num_values = std::count(line.begin(), line.end(), separator) + 1;
    if (expected_num_values != actual_num_values) {
        std::string error = "Error: Expected " + std::to_string(expected_num_values) +
                            " values while reading file; but " + std::to_string(actual_num_values) + " were found";
        throw std::runtime_error(error);
    }
    // Separate the line into its components, now that the number of values has been checked
    std::size_t parsed_line_size = parsed_line.size();
    // Make a copy that we can modify
    std::string line_copy = line;
    for (std::size_t n = 0; n < parsed_line_size - 1; n++) {
        std::size_t pos = line_copy.find(separator);
        parsed_line[n] = line_copy.substr(0, pos);
        line_copy = line_copy.substr(pos + 1, std::string::npos);
    }
    parsed_line[parsed_line_size - 1] = line_copy;
}

// Reads a line from an input file stream and outputs the components of the line split at "separator" (spaces used by
// default). By default, expects 4 components of the line to separate (used in parsing vtk header data)
std::vector<std::string> splitString(std::ifstream &input_data_stream, std::size_t expected_num_values,
                                     char separator) {

    std::string line;
    std::vector<std::string> parsed_line(expected_num_values);
    getline(input_data_stream, line);
    splitString(line, parsed_line, expected_num_values, separator);
    return parsed_line;
}

bool checkFileExists(const std::string path, const int id, const bool error) {
    std::ifstream stream;
    stream.open(path);
    if (!(stream.is_open())) {
        stream.close();
        if (error)
            throw std::runtime_error("Could not locate/open \"" + path + "\"");
        else
            return false;
    }
    stream.close();
    if (id == 0)
        std::cout << "Opened \"" << path << "\"" << std::endl;
    return true;
}

std::string checkFileInstalled(const std::string name, const int id) {
    // Path to file. Prefer installed location; if not installed use source location.
    std::string path = ExaCA_DATA_INSTALL;
    std::string file = path + "/" + name;
    bool files_installed = checkFileExists(file, id, false);
    if (!files_installed) {
        // If full file path, just use it.
        if (name.substr(0, 1) == "/") {
            file = name;
        }
        // If a relative path, it has to be with respect to the source path.
        else {
            path = ExaCA_DATA_SOURCE;
            file = path + "/" + name;
        }
        checkFileExists(file, id);
    }
    return file;
}

// Make sure file contains data
void checkFileNotEmpty(std::string testfilename) {
    std::ifstream testfilestream;
    testfilestream.open(testfilename);
    std::string testline;
    std::getline(testfilestream, testline);
    if (testline.empty())
        throw std::runtime_error("First line of file " + testfilename + " appears empty");
    testfilestream.close();
}

// Check if the temperature data is in ASCII or binary format
bool checkTemperatureFileFormat(std::string tempfile_thislayer) {
    bool binary_input_data;
    std::size_t found = tempfile_thislayer.find(".catemp");
    if (found == std::string::npos)
        binary_input_data = false;
    else
        binary_input_data = true;
    return binary_input_data;
}

// Check to make sure that the 6 expected column names appear in the correct order in the header for this temperature
// file Return the number of columns present - ignore any columns after the 6 of interest
std::size_t checkForHeaderValues(std::string header_line) {

    // Header values from file - number of commas plus one is the size of the header
    std::size_t header_size = std::count(header_line.begin(), header_line.end(), ',') + 1;
    std::vector<std::string> header_values(header_size, "");
    splitString(header_line, header_values, header_size);

    std::vector<std::vector<std::string>> expected_values = {
        {"x",  "x(m)"},
        {"y",  "y(m)"},
        {"z",  "z(m)"},
        {"tm", "tm(s)"},
        {"tl", "tl(s)", "ts", "ts(s)"},
        {"r",  "r(k/s)", "cr", "cr(k/s)"}
    };
    std::size_t num_expected_values = expected_values.size();
    if (num_expected_values > header_size)
        throw std::runtime_error("Error: Fewer values than expected found in temperature file header");

    // Case insensitive comparison
    for (std::size_t n = 0; n < num_expected_values; n++) {
        auto val = removeWhitespace(header_values[n]);
        std::transform(val.begin(), val.end(), val.begin(), ::tolower);
        // Check each header column label against the expected value(s) - throw error if no match
        std::size_t options_size = expected_values[n].size();
        for (std::size_t e = 0; e < options_size; e++) {
            auto ev = expected_values[n][e];
            if (val == ev)
                break;
            else if (e == options_size - 1)
                throw std::runtime_error(ev + " not found in temperature file header");
        }
    }
    return header_size;
}

// Read and discard "n_lines" lines of data from the file
void skipLines(std::ifstream &input_data_stream, const int n_lines) {
    std::string dummy_str;
    for (int line = 0; line < n_lines; line++)
        getline(input_data_stream, dummy_str);
}
