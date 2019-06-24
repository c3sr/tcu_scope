#!/usr/bin/env python3

# Copyright (c) 2013, 2015, Ruslan Baratov
# All rights reserved.
####################################################
# Example Usage:
# python3 ./tools/generate_sugar_files.py --top `pwd`/src --var BENCHMARK --test BENCHMARK --exclude-dirs _resources
####################################################

import argparse
import os
import re
import sys

wiki = 'https://github.com/ruslo/sugar/wiki/Collecting-sources'
path = os.path.dirname(os.path.abspath(__file__))
base = os.path.basename(__file__)


def first_is_subdirectory_of_second(subdir_name, dir_name):
    subdir_name = subdir_name.rstrip(os.sep)
    dir_name = dir_name.rstrip(os.sep)
    if subdir_name == dir_name:
        return True
    if not subdir_name.startswith(dir_name):
        return False
    rest = subdir_name[len(dir_name):]
    if rest.startswith(os.sep):
        return True
    return False


class Generator:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Generate sugar.cmake files according to directory struct'
        )
        # print(os.path.join(path, '..', 'src'))
        self.exclude_dirs = []

    def parse(self):
        self.parser.add_argument(
            '--top',
            type=str,
            default=os.path.join(path, '..', 'src'),
            required=False,
            help='top directory of sources'
        )

        self.parser.add_argument(
            '--var',
            type=str,
            required=False,
            default='BENCHMARK',
            help='variable name'
        )

        self.parser.add_argument(
            '--test',
            type=str,
            required=False,
            default='BENCHMARK',
            help='Tests variable name'
        )

        self.parser.add_argument(
            '--exclude-dirs',
            type=str,
            nargs='*',
            default=['_resources', 'cuew'],
            help='Ignore this directories'
        )

        self.parser.add_argument(
            '--exclude-filenames',
            type=str,
            nargs='*',
            default=[],
            help='Ignore this filenames'
        )

    def make_header_guard(dir):
        dir = dir.upper()
        dir = re.sub(r'\W', '_', dir)
        dir = re.sub('_+', '_', dir)
        dir = dir.lstrip('_')
        dir = dir.rstrip('_')
        dir += '_'
        return dir

    def process_file(relative, source_variable, tests_variable, file_name, filelist, dirlist):
        def is_header(f):
            return f.endswith(".hpp") or f.endswith(".h") or f.endswith(".hxx") or f.endswith(".hh") or f.endswith(".cuh")

        def is_source(f):
            return f.endswith(".cpp") or f.endswith(".cc") or f.endswith(".c") or f.endswith(".cxx")

        def is_cuda_header(f):
            return f.endswith(".cuh")

        def is_cuda_source(f):
            return f.endswith(".cu")

        cpp_or_hpp_files = [
            f for f in filelist if is_header(f) or is_cuda_header(f) or is_source(f) or is_cuda_source(f)]
        if cpp_or_hpp_files == []:
            return
        with open(file_name, 'w') as file_id:
            file_id.write(
                '# This file generated automatically by:\n'
                '#   {}\n'
                '# see wiki for more info:\n'
                '#   {}\n\n'.format(base, wiki)
            )
            relative += '/sugar.cmake'
            hg = Generator.make_header_guard(relative)
            file_id.write(
                'if(DEFINED {})\n'
                '  return()\n'
                'else()\n'
                '  set({} 1)\n'
                'endif()\n\n'.format(hg, hg)
            )
            if filelist:
                file_id.write('include(sugar_files)\n')
            if dirlist:
                file_id.write('include(sugar_include)\n')
            if filelist or dirlist:
                file_id.write('\n')

            if dirlist:
                for x in dirlist:
                    file_id.write("sugar_include({})\n".format(x))
                file_id.write('\n')

            if filelist:
                files = [f for f in filelist if not f.endswith(
                    "_test.cpp") and is_header(f) and not is_cuda_header(f)]
                if files != []:
                    file_id.write("sugar_files(\n")
                    # file_id.write("    {}\n".format(hg + "SOURCES"))
                    file_id.write("    {}\n".format(
                        source_variable + "_HEADERS"))
                    for x in files:
                        file_id.write("    {}\n".format(x))
                    file_id.write(")\n")
                    file_id.write('\n')
            if filelist:
                files = [f for f in filelist if not f.endswith(
                    "_test.cpp") and is_cuda_header(f)]
                if files != []:
                    file_id.write("sugar_files(\n")
                    # file_id.write("    {}\n".format(hg + "SOURCES"))
                    file_id.write("    {}\n".format(
                        source_variable + "_CUDA_HEADERS"))
                    for x in files:
                        file_id.write("    {}\n".format(x))
                    file_id.write(")\n")
                    file_id.write('\n')
            if filelist:
                files = [f for f in filelist if not f.endswith(
                    "_test.cpp") and is_source(f)]
                if files != []:
                    file_id.write("sugar_files(\n")
                    # file_id.write("    {}\n".format(hg + "SOURCES"))
                    file_id.write("    {}\n".format(
                        source_variable + "_SOURCES"))
                    for x in files:
                        file_id.write("    {}\n".format(x))
                    file_id.write(")\n")
                    file_id.write('\n')
            if filelist:
                files = [f for f in filelist if not f.endswith(
                    "_test.cpp") and is_cuda_source(f)]
                if files != []:
                    file_id.write("sugar_files(\n")
                    # file_id.write("    {}\n".format(hg + "SOURCES"))
                    file_id.write("    {}\n".format(
                        source_variable + "_CUDA_SOURCES"))
                    for x in files:
                        file_id.write("    {}\n".format(x))
                    file_id.write(")\n")
                    file_id.write('\n')

            if filelist:
                files = [f for f in filelist if f.endswith("_test.cpp")]
                if files != []:
                    file_id.write("sugar_files(\n")
                    #file_id.write("    {}\n".format(hg + "TESTS"))
                    file_id.write("    {}\n".format(
                        tests_variable + "_TEST_SOURCES"))
                    for x in files:
                        file_id.write("    {}\n".format(x))
                    file_id.write(")\n")

    def is_excluded(self, dir_name):
        for x in self.exclude_dirs:
            if first_is_subdirectory_of_second(dir_name, x):
                return True
            if os.path.basename(dir_name) == x:
                return True
        return False

    def create(self):
        args = self.parser.parse_args()

        cwd = os.getcwd()
        for x in args.exclude_dirs:
            self.exclude_dirs.append(x)
        if args.exclude_filenames:
            exclude_filenames = args.exclude_filenames
        else:
            exclude_filenames = []
        exclude_filenames += ['sugar.cmake', 'CMakeLists.txt', '.DS_Store']

        source_variable = args.var
        tests_variable = args.test
        for rootdir, dirlist, filelist in os.walk(args.top):
            for x in exclude_filenames:
                try:
                    filelist.remove(x)
                except ValueError:
                    pass  # ignore if not in list

            rootdir = os.path.abspath(rootdir)

            if self.is_excluded(rootdir):
                continue

            new_dirlist = []
            for x in dirlist:
                x_abs = os.path.join(rootdir, x)
                if not self.is_excluded(x_abs):
                    new_dirlist.append(x)

            relative = os.path.relpath(rootdir, cwd)
            file_name = '{}/sugar.cmake'.format(rootdir)
            Generator.process_file(
                relative, source_variable,
                tests_variable,
                file_name, filelist, new_dirlist
            )

    def run():
        generator = Generator()
        generator.parse()
        generator.create()

if __name__ == '__main__':
    Generator.run()
