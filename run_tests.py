#!/usr/bin/env python3
"""Test runner script with multiple execution modes.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --unit       # Run unit tests only
    python run_tests.py --integration # Run integration tests only
    python run_tests.py --quick      # Run quick tests only
    python run_tests.py -v           # Verbose output
    python run_tests.py --coverage   # Generate coverage report
"""

import subprocess
import sys
import argparse


def run_unit_tests(verbose=False):
    """Run unit tests only.

    Args:
        verbose: Enable verbose output

    Returns:
        subprocess.CompletedProcess result
    """
    cmd = ["pytest", "tests/unit/", "-v" if verbose else "-q"]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd)


def run_integration_tests(verbose=False):
    """Run integration tests only.

    Args:
        verbose: Enable verbose output

    Returns:
        subprocess.CompletedProcess result
    """
    cmd = ["pytest", "tests/integration/", "-v" if verbose else "-q"]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd)


def run_edge_case_tests(verbose=False):
    """Run edge case tests only.

    Args:
        verbose: Enable verbose output

    Returns:
        subprocess.CompletedProcess result
    """
    cmd = ["pytest", "tests/edge_cases/", "-v" if verbose else "-q"]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd)


def run_all_tests(verbose=False, coverage=False):
    """Run all tests.

    Args:
        verbose: Enable verbose output
        coverage: Generate coverage report

    Returns:
        subprocess.CompletedProcess result
    """
    cmd = ["pytest", "tests/"]
    if verbose:
        cmd.append("-v")
    if coverage:
        cmd.extend([
            "--cov=.",
            "--cov-report=html",
            "--cov-report=term"
        ])
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd)


def run_quick_tests():
    """Run quick tests only (skip slow/integration).

    Returns:
        subprocess.CompletedProcess result
    """
    cmd = ["pytest", "tests/unit/", "-q", "-m", "not slow"]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd)


def run_specific_test_file(test_file, verbose=False):
    """Run a specific test file.

    Args:
        test_file: Path to test file
        verbose: Enable verbose output

    Returns:
        subprocess.CompletedProcess result
    """
    cmd = ["pytest", test_file, "-v" if verbose else "-q"]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd)


def list_all_tests():
    """List all test files without running them.

    Returns:
        subprocess.CompletedProcess result
    """
    cmd = ["pytest", "tests/", "--collect-only", "-q"]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd)


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Run DSPy QA tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    Run all tests
  python run_tests.py --unit             Run unit tests only
  python run_tests.py --integration       Run integration tests only
  python run_tests.py --edge-cases        Run edge case tests only
  python run_tests.py --quick             Run quick tests only
  python run_tests.py -v                  Verbose output
  python run_tests.py --coverage          Generate coverage report
  python run_tests.py --list              List all tests without running
  python run_tests.py tests/unit/test_qa_module.py  Run specific test file
        """
    )

    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests only"
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests only"
    )
    parser.add_argument(
        "--edge-cases",
        action="store_true",
        help="Run edge case tests only"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only (skip slow tests)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report (HTML and terminal)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all tests without running them"
    )
    parser.add_argument(
        "test_file",
        nargs="?",
        help="Specific test file to run"
    )

    args = parser.parse_args()

    # Print header
    print("=" * 60)
    print("DSPy QA Test Runner")
    print("=" * 60)
    print()

    # Determine which test suite to run
    if args.list:
        result = list_all_tests()
    elif args.test_file:
        result = run_specific_test_file(args.test_file, args.verbose)
    elif args.unit:
        print("Running unit tests...")
        print()
        result = run_unit_tests(verbose=args.verbose)
    elif args.integration:
        print("Running integration tests...")
        print()
        result = run_integration_tests(verbose=args.verbose)
    elif args.edge_cases:
        print("Running edge case tests...")
        print()
        result = run_edge_case_tests(verbose=args.verbose)
    elif args.quick:
        print("Running quick tests...")
        print()
        result = run_quick_tests()
    else:
        print("Running all tests...")
        print()
        result = run_all_tests(verbose=args.verbose, coverage=args.coverage)

    # Print coverage report location if generated
    if args.coverage and result.returncode == 0:
        print()
        print("=" * 60)
        print("Coverage report generated: htmlcov/index.html")
        print("=" * 60)

    # Exit with appropriate code
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
