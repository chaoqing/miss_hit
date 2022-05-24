#!/usr/bin/env python3
##############################################################################
##                                                                          ##
##          MATLAB Independent, Small & Safe, High Integrity Tools          ##
##                                                                          ##
##              Copyright (C) 2019-2022, Florian Schanda                    ##
##                                                                          ##
##  This file is part of MISS_HIT.                                          ##
##                                                                          ##
##  MATLAB Independent, Small & Safe, High Integrity Tools (MISS_HIT) is    ##
##  free software: you can redistribute it and/or modify it under the       ##
##  terms of the GNU General Public License as published by the Free        ##
##  Software Foundation, either version 3 of the License, or (at your       ##
##  option) any later version.                                              ##
##                                                                          ##
##  MISS_HIT is distributed in the hope that it will be useful,             ##
##  but WITHOUT ANY WARRANTY; without even the implied warranty of          ##
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           ##
##  GNU General Public License for more details.                            ##
##                                                                          ##
##  You should have received a copy of the GNU General Public License       ##
##  along with MISS_HIT. If not, see <http://www.gnu.org/licenses/>.        ##
##                                                                          ##
##############################################################################

# Main test driver

import os
import sys
import copy
import subprocess
import multiprocessing
import argparse
import shutil

TEST_ROOT = os.getcwd()
MH_ROOT = os.path.normpath(os.path.join(TEST_ROOT, ".."))
TEST_ENV = copy.copy(os.environ)
TEST_ENV["PYTHONIOENCODING"] = "UTF-8"
TEST_ENV["PYTHONPATH"] = MH_ROOT

CBMC_AVILABLE=shutil.which("cbmc") is not None


def run_command(command, args):
    cmd = ["coverage",
           "run",
           "--rcfile=%s" % os.path.join(TEST_ROOT, "coverage.cfg"),
           "--branch",
           "--append"]
    cmd.append(os.path.join("..", "..", "..", command))
    cmd += args

    rv = subprocess.run(cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        encoding="utf-8",
                        env=TEST_ENV)

    return rv


def run_module(module, args):
    cmd = ["coverage",
           "run",
           "--rcfile=%s" % os.path.join(TEST_ROOT, "coverage.cfg"),
           "--branch",
           "--append",
           "-m",
           module]
    cmd += args

    rv = subprocess.run(cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        encoding="utf-8",
                        env=TEST_ENV)

    return rv


def relevant_files():
    rv = []
    for path, _, files in os.walk("."):
        for f in files:
            if f.endswith(".m") or f.endswith(".slx"):
                rv.append(os.path.join(path, f))
    return rv


def backup_files(files):
    originals = {}
    for f in files:
        with open(f, "rb") as fd:
            originals[f] = fd.read()
    return originals


def rename_as_fixed(files):
    fixed = {}
    for f in files:
        fixed_name = f + "_fixed"
        with open(f, "rb") as fd:
            fixed[f] = fd.read()
        with open(fixed_name, "wb") as fd:
            fd.write(fixed[f])
    return fixed


def detect_broken_fixes(files, fixed):
    broken_fixes = set()
    for f in files:
        with open(f, "rb") as fd:
            tmp = fd.read()
        if tmp != fixed[f]:
            broken_fixes.add(f)
    return broken_fixes


def restore_originals(files, originals):
    for f in files:
        with open(f, "wb") as fd:
            fd.write(originals[f])


def write_simulink_diffs(files):
    for f in files:
        if f.endswith(".slx"):
            r = run_command("mh_diff",
                            ["--allow-weird-names",
                             f,
                             f + "_fixed"])
            with open(f + ".diff", "w") as fd:
                fd.write(r.stdout)


def execute_style_test(name):
    files = relevant_files()
    original_content = backup_files(files)

    # Run in HTML mode
    r = run_command("mh_style",
                    [".",
                     "--single",
                     "--process-slx",
                     "--html=expected_out.html"])
    html_out = r.stdout

    # Run in plaintext mode and fix
    r = run_command("mh_style",
                    [".",
                     "--debug-validate-links",
                     "--single",
                     "--process-slx",
                     "--fix"])
    plain_out = r.stdout

    # Write the fixed file to foo.m_fixed
    fixed_content = rename_as_fixed(files)

    # Run in plaintext mode, again, to see if more things need fixing
    r = run_command("mh_style",
                    [".",
                     "--single",
                     "--process-slx",
                     "--fix"])
    plain_out_again = r.stdout

    # Check if fixed files not "fixed" again, and then restore
    # originals
    broken_fixes = detect_broken_fixes(files, fixed_content)
    restore_originals(files, original_content)

    # Save stdout
    with open("expected_out.txt", "w") as fd:
        fd.write("=== PLAIN MODE ===\n")
        fd.write(plain_out)
        fd.write("\n")
        fd.write("=== HTML MODE ===\n")
        fd.write(html_out)
        if broken_fixes:
            fd.write("\n")
            fd.write("=== ! BROKEN FIXES ! ===\n")
            for fail in sorted(broken_fixes):
                fd.write("Fixing is not idempotent for %s\n" % fail)

    # Write diff for Simulink files
    write_simulink_diffs(files)

    return "Ran style test %s" % name


def execute_metric_test(name):
    # Run
    r = run_command("mh_metric",
                    [".",
                     "--single"])
    plain_out = r.stdout

    # HTML
    r = run_command("mh_metric",
                    [".",
                     "--single",
                     "--portable-html",
                     "--html=metrics.html"])
    html_out = r.stdout

    # JSON
    r = run_command("mh_metric",
                    [".",
                     "--single",
                     "--json=metrics.json"])
    json_out = r.stdout

    # Save stdout
    with open("expected_out.txt", "w") as fd:
        fd.write("=== PLAIN MODE ===\n")
        fd.write(plain_out)

        fd.write("\n\n=== HTML MODE ===\n")
        fd.write(html_out)

        fd.write("\n\n=== JSON MODE ===\n")
        fd.write(json_out)

    return "Ran metrics test %s" % name


def execute_lint_test(name):
    # Run
    r = run_command("mh_lint",
                    [".",
                     "--single"])
    plain_out = r.stdout

    # # HTML
    # r = subprocess.run([sys.executable,
    #                     "../../../mh_lint",
    #                     "--single",
    #                     "--html=lint.html",
    #                     ".",],
    #                    stdout=subprocess.PIPE,
    #                    stderr=subprocess.STDOUT,
    #                    encoding="utf-8",
    #                    env=TEST_ENV)
    # html_out = r.stdout

    # Save stdout
    with open("expected_out.txt", "w") as fd:
        fd.write("=== PLAIN MODE ===\n")
        fd.write(plain_out)

        # fd.write("\n\n=== HTML MODE ===\n")
        # fd.write(html_out)

    return "Ran lint test %s" % name


def execute_lexer_test(name):
    files = [f
             for f in os.listdir(".")
             if f.endswith(".m")]

    for f in files:
        r = run_module("miss_hit_core.m_lexer", [f])
        plain_out = r.stdout

        with open(f + ".out", "w") as fd:
            fd.write(plain_out)

    return "Ran lexer test %s" % name


def execute_parser_test(name):
    files = [f
             for f in os.listdir(".")
             if f.endswith(".m")]

    for f in files:
        r = run_command("mh_debug_parser",
                        ["--no-tb",
                         "--tree",
                         f])
        plain_out_matlab = r.stdout

        r = run_command("mh_debug_parser",
                        ["--no-tb",
                         "--tree",
                         "--octave",
                         f])
        plain_out_octave = r.stdout

        with open(f + ".out", "w") as fd:
            fd.write(plain_out_matlab)
        if plain_out_matlab != plain_out_octave:
            with open(f + ".octave.out", "w") as fd:
                fd.write(plain_out_octave)


    return "Ran parser test %s" % name


def execute_sem_test(name):
    r = run_command("mh_lint",
                    ["--entry-point=test",
                     "--debug-show-global-symbol-table",
                     "--debug-show-path"])
    plain_out = r.stdout

    with open("test.out", "w") as fd:
        fd.write(plain_out)

    return "Ran sem test %s" % name


def execute_simulink_parser_test(name):
    files = [f
             for f in os.listdir(".")
             if f.endswith(".slx")]

    for f in files:
        r = run_module("miss_hit_core.s_parser", [f])
        plain_out = r.stdout

        with open(f + ".out", "w") as fd:
            fd.write(plain_out)

    return "Ran simulink parser test %s" % name


def execute_config_parser_test(name):
    files = [f
             for f in os.listdir(".")
             if f.endswith(".cfg")]

    for f in files:
        r = run_module("miss_hit_core.cfg_parser",
                       ["--no-tb",
                        f])
        plain_out = r.stdout

        with open(f + ".out", "w") as fd:
            fd.write(plain_out)

    return "Ran config parser test %s" % name


def execute_sanity_test(name):
    if os.path.isfile(os.path.join(MH_ROOT, "miss_hit_core", name + ".py")):
        module = "miss_hit_core"
    elif os.path.isfile(os.path.join(MH_ROOT, "miss_hit", name + ".py")):
        module = "miss_hit"
    else:
        return "FAILED sanity test %s (cannot find module)" % name

    r = run_module("%s.%s" % (module, name), [])
    plain_out = r.stdout

    with open("%s.out" % name, "w") as fd:
        fd.write(plain_out.rstrip() + "\n")

    return "Ran sanity test %s" % name


def execute_trace_test(name):
    for filename in ("mh_trace.json", "mh_trace_by_tag.json"):
        if os.path.isfile(filename):
            os.unlink(filename)

    flags = []
    if os.path.isfile("cmdline"):
        with open("cmdline", "r") as fd:
            for raw_flag in fd.readlines():
                flag = raw_flag.strip()
                if flag:
                    flags.append(flag)

    r = run_command("mh_trace", flags)
    plain_out = r.stdout
    with open("test.out", "w") as fd:
        fd.write(plain_out)

    r = run_command("mh_trace",
                    flags +
                    ["--json=mh_trace_by_tag.json",
                     "--by-tag"])

    return "Ran trace test %s" % name


def execute_bmc_test(name):
    m_files = []
    for path, _, files in os.walk("."):
        for f in files:
            if f.endswith(".m"):
                m_files.append(os.path.join(path, f))

    for filename in m_files:
        file_root = os.path.splitext(filename)[0]

        r = run_command("mh_bmc",
                        ["--single",
                         filename])
        plain_out = r.stdout

        with open("%s.txt" % file_root, "w") as fd:
            fd.write(plain_out.rstrip() + "\n")

    return "Ran bmc test %s" % name


def execute_project_test(name):
    flags = ["--single"]
    if os.path.isfile("cmdline"):
        with open("cmdline", "r") as fd:
            for raw_flag in fd.readlines():
                flag = raw_flag.strip()
                if flag:
                    flags.append(flag)

    with open("output.txt", "w") as fd:
        fd.write("=== STYLE ===\n")
        r = run_command("mh_style", flags)
        plain_out = r.stdout
        fd.write(plain_out.rstrip() + "\n")

        fd.write("=== LINT ===\n")
        r = run_command("mh_lint", flags)
        plain_out = r.stdout
        fd.write(plain_out.rstrip() + "\n")

        fd.write("=== METRICS ===\n")
        r = run_command("mh_metric", flags)
        plain_out = r.stdout
        fd.write(plain_out.rstrip() + "\n")

        fd.write("=== TRACING ===\n")
        r = run_command("mh_trace", flags)
        plain_out = r.stdout
        fd.write(plain_out.rstrip() + "\n")

    return "Ran project test %s" % name


def execute_copyright_test(name):
    files = relevant_files()
    original_content = backup_files(files)

    flags = ["--single", "--process-slx"]
    if os.path.isfile("cmdline"):
        with open("cmdline", "r") as fd:
            for raw_flag in fd.readlines():
                flag = raw_flag.strip()
                if flag:
                    flags.append(flag)

    # Run in plaintext mode and fix
    r = run_command("mh_copyright", flags)
    plain_out = r.stdout

    # Write the fixed file to foo.m_fixed
    fixed_content = rename_as_fixed(files)

    # Run in plaintext mode, again, to see if more things need fixing
    r = run_command("mh_copyright", flags)
    plain_out_again = r.stdout

    # Check if fixed files not "fixed" again, and then restore
    # originals
    broken_fixes = detect_broken_fixes(files, fixed_content)
    restore_originals(files, original_content)

    # Save stdout
    with open("expected_out.txt", "w") as fd:
        fd.write("=== PLAIN MODE ===\n")
        fd.write(plain_out)
        if broken_fixes:
            fd.write("\n")
            fd.write("=== ! BROKEN FIXES ! ===\n")
            for fail in sorted(broken_fixes):
                fd.write("Fixing is not idempotent for %s\n" % fail)

    # Write diff for Simulink files
    write_simulink_diffs(files)

    return "Ran copyright test %s" % name


def run_test(test):
    if os.path.exists(os.path.join(TEST_ROOT,
                                   test["kind"],
                                   test["test"],
                                   "ONLY_LINUX")):
        if sys.platform != "linux":
            return "SKIPPED linux-only test %s" % test["test"]
    elif test["kind"] == "bmc" and not CBMC_AVILABLE:
        return "SKIPPED BMC test %s" % test["test"]

    # Set up in the correct directory
    os.chdir(os.path.join(TEST_ROOT,
                          test["kind"],
                          test["test"]))

    # Execute test
    fn = {
        "style"           : execute_style_test,
        "metrics"         : execute_metric_test,
        "lint"            : execute_lint_test,
        "trace"           : execute_trace_test,
        "bmc"             : execute_bmc_test,
        "lexer"           : execute_lexer_test,
        "parser"          : execute_parser_test,
        "sem"             : execute_sem_test,
        "simulink_parser" : execute_simulink_parser_test,
        "config_parser"   : execute_config_parser_test,
        "sanity"          : execute_sanity_test,
        "projects"        : execute_project_test,
        "copyright"       : execute_copyright_test,
    }
    test_result = fn[test["kind"]](test["test"])

    # Move coverage results
    if os.path.isfile(".coverage"):
        os.rename(".coverage", os.path.join(TEST_ROOT,
                                            ".".join([".coverage",
                                                      test["kind"],
                                                      test["test"]])))

    return test_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single",
                    action="store_true",
                    default=False)
    ap.add_argument("--suite",
                    default=None)
    ap.add_argument("--no-summary",
                    default=False,
                    action="store_true")
    ap.add_argument("--no-clear",
                    default=False,
                    action="store_true")

    options = ap.parse_args()

    # Make sure we're in the right directory
    assert os.path.isfile("../mh_style")
    assert os.path.isdir("../miss_hit_core")
    root = os.getcwd()

    tests = []

    if options.suite:
        suites = [options.suite]
    else:
        suites = ["lexer", "parser", "simulink_parser", "sem",
                  "config_parser",
                  "style", "metrics", "lint", "trace", "bmc", "copyright",
                  "projects",
                  "sanity"]

    for kind in suites:
        for t in os.listdir(kind):
            tests.append({"kind" : kind,
                          "test" : t})

    if not options.no_clear:
        os.system("coverage erase")

    if options.single:
        for t in tests:
            print("Running %s" % t)
            print(run_test(t))
    else:
        pool = multiprocessing.Pool()
        for res in pool.imap_unordered(run_test, tests, 2):
            print(res)

    os.chdir(TEST_ROOT)
    if options.no_clear:
        os.system("coverage combine --append")
    else:
        os.system("coverage combine")

    if not options.no_summary:
        os.system("coverage html --rcfile=coverage.cfg")
        os.system("coverage report --rcfile=coverage.cfg")


if __name__ == "__main__":
    main()
