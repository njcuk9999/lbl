#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2026-02-01 at 13:20

@author: cook
"""
import ast
from typing import Dict

from lbl.recipes import (lbl_compute, lbl_template, lbl_mask, lbl_find,
                         lbl_setup, lbl_compile, lbl_noise, lbl_wrap,
                         lbl_reset, lbl_resmap, lbl_telluclean, lbl_demo)


# =============================================================================
# Define variables
# =============================================================================
apero_input = """
2026-02-01 10:55:25.774|I|LBLCOMPUTE_FP[00390]| User keyword arguments:
2026-02-01 10:55:25.775|I|LBLCOMPUTE_FP[00390]| 	OBJECT_SCIENCE="FP"
2026-02-01 10:55:25.775|I|LBLCOMPUTE_FP[00390]| 	OBJECT_TEMPLATE="FP"
2026-02-01 10:55:25.776|I|LBLCOMPUTE_FP[00390]| 	DATA_TYPE="FP"
2026-02-01 10:55:25.777|I|LBLCOMPUTE_FP[00390]| 	INSTRUMENT="SPIROU"
2026-02-01 10:55:25.778|I|LBLCOMPUTE_FP[00390]| 	DATA_DIR="/scratch2/spirou/drs-data/spirou_xxs_08/lbl"
2026-02-01 10:55:25.778|I|LBLCOMPUTE_FP[00390]| 	DATA_SOURCE="APERO"
2026-02-01 10:55:25.779|I|LBLCOMPUTE_FP[00390]| 	SKIP_DONE="True"
2026-02-01 10:55:25.779|I|LBLCOMPUTE_FP[00390]| 	ITERATION="0"
2026-02-01 10:55:25.780|I|LBLCOMPUTE_FP[00390]| 	TOTAL="5"
2026-02-01 10:55:25.781|I|LBLCOMPUTE_FP[00390]| 	PROGRAM="LBLCOMPUTE_FP[00390]"
2026-02-01 10:55:25.781|I|LBLCOMPUTE_FP[00390]| 	RESPROJ_TABLES="{'DTEMP3000': 'temperature_gradient_3000.fits', 'DTEMP3500': 'temperature_gradient_3500.fits', 'DTEMP4000': 'temperature_gradient_4000.fits', 'DTEMP4500': 'temperature_gradient_4500.fits', 'DTEMP5000': 'temperature_gradient_5000.fits', 'DTEMP5500': 'temperature_gradient_5500.fits', 'DTEMP6000': 'temperature_gradient_6000.fits'}"
"""
apero_input = """
2026-02-01 10:55:25.774|I|LBLTEMPLATE_FP[00390]| User keyword arguments:
2026-02-01 10:55:25.775|I|LBLTEMPLATE_FP[00390]| 	OBJECT_SCIENCE="FP"
2026-02-01 10:55:25.775|I|LBLTEMPLATE_FP[00390]| 	OBJECT_TEMPLATE="FP"
2026-02-01 10:55:25.776|I|LBLTEMPLATE_FP[00390]| 	DATA_TYPE="FP"
2026-02-01 10:55:25.777|I|LBLTEMPLATE_FP[00390]| 	INSTRUMENT="SPIROU"
2026-02-01 10:55:25.778|I|LBLTEMPLATE_FP[00390]| 	DATA_DIR="/scratch2/spirou/drs-data/spirou_xxs_08/lbl"
2026-02-01 10:55:25.778|I|LBLTEMPLATE_FP[00390]| 	DATA_SOURCE="APERO"
2026-02-01 10:55:25.779|I|LBLTEMPLATE_FP[00390]| 	SKIP_DONE="False"
2026-02-01 10:55:25.779|I|LBLTEMPLATE_FP[00390]| 	OVERWRITE="True"
2026-02-01 10:55:25.779|I|LBLTEMPLATE_FP[00390]| 	ITERATION="0"
2026-02-01 10:55:25.780|I|LBLTEMPLATE_FP[00390]| 	TOTAL="5"
2026-02-01 10:55:25.781|I|LBLTEMPLATE_FP[00390]| 	PROGRAM="LBLTEMPLATE_FP[00390]"
2026-02-01 10:55:25.781|I|LBLTEMPLATE_FP[00390]| 	RESPROJ_TABLES="{'DTEMP3000': 'temperature_gradient_3000.fits', 'DTEMP3500': 'temperature_gradient_3500.fits', 'DTEMP4000': 'temperature_gradient_4000.fits', 'DTEMP4500': 'temperature_gradient_4500.fits', 'DTEMP5000': 'temperature_gradient_5000.fits', 'DTEMP5500': 'temperature_gradient_5500.fits', 'DTEMP6000': 'temperature_gradient_6000.fits'}"
"""
# -----------------------------------------------------------------------------

# =============================================================================
# Define functions
# =============================================================================
def _is_valid_int(value_str):
    """Check if string is a valid integer without raising exceptions."""
    if not value_str or '.' in value_str:
        return False

    # Remove leading sign if present
    check_str = value_str[1:] if value_str[0] in ('+', '-') else value_str

    # Check if all remaining characters are digits
    return check_str.isdigit() and len(check_str) > 0


def _is_valid_float(value_str):
    """Check if string is a valid float without raising exceptions."""
    if not value_str or value_str.count('.') != 1:
        return False

    # Remove leading sign if present
    check_str = value_str[1:] if value_str[0] in ('+', '-') else value_str

    # Split by decimal point and check both parts are digits
    parts = check_str.split('.')
    return len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()


def _is_valid_dict_or_list(value_str):
    """Check if string looks like a valid dict or list by counting brackets."""
    if value_str.startswith('{') and value_str.endswith('}'):
        # Basic dict check: equal number of { and }
        return value_str.count('{') == value_str.count('}')
    elif value_str.startswith('[') and value_str.endswith(']'):
        # Basic list check: equal number of [ and ]
        return value_str.count('[') == value_str.count(']')
    return False


def _parse_literal(value_str):
    """Parse dict/list literals. Assumes validation has already been done."""
    return ast.literal_eval(value_str)


def safe_cast_value(value_str):
    """
    Safely cast a string value to its appropriate Python type.
    Handles: dicts, lists, ints, floats, bools, and strings.
    NO EXCEPTIONS RAISED - uses validation instead of try/except.

    Args:
        value_str (str): String representation of a value

    Returns:
        Casted value in appropriate type, or original string if casting fails
    """
    value_str = value_str.strip()

    # Handle booleans
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False

    # Handle None
    if value_str.lower() == 'none':
        return None

    # Handle dicts and lists - validate structure first
    if _is_valid_dict_or_list(value_str):
        return _parse_literal(value_str)

    # Handle integers
    if _is_valid_int(value_str):
        return int(value_str)

    # Handle floats
    if _is_valid_float(value_str):
        return float(value_str)

    # Return as string if no conversion matched
    return value_str


def parse_lbl_log(log_input):
    """
    Parse LBL log input and extract recipe name and arguments.

    Args:
        log_input (str): Log string containing PROGRAM and keyword arguments

    Returns:
        tuple: (recipe_name, arguments_dict)
    """
    lines = log_input.strip().split('\n')

    recipe_name = None
    arguments = {}

    for line in lines:
        # Extract PROGRAM name
        if 'PROGRAM=' in line:
            # Extract the program name (e.g., "LBLCOMPUTE_FP" from "LBLCOMPUTE_FP[00390]")
            parts = line.split('PROGRAM="')
            if len(parts) > 1:
                program = parts[1].split('"')[0]
                # Extract recipe name (remove timestamp info like [00390])
                program = program.split('[')[0]  # Remove [00390]
                # split by '_' and take first part for recipe name
                program = program.split('_')[0]
                # Convert to lowercase and remove trailing numbers
                recipe_name = program.lower()

    for line in lines:
        # Extract key=value pairs
        if '=' in line and 'PROGRAM' not in line:
            # Extract the key=value part
            match = line.split('| ')[-1].strip()
            if '=' in match:
                key, value = match.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"')
                # Safely cast the value to appropriate type
                value = safe_cast_value(value)
                arguments[key] = value

    return recipe_name, arguments


def run_lbl_from_log(recipe_name: str, arguments: Dict[str, str]):
    """
    Convert parsed log arguments to an LBL run by calling the appropriate recipe.

    Args:
        recipe_name (str): Name of the recipe to run (e.g., 'lblcompute')
        arguments (Dict[str, str]): Parsed arguments dictionary
    """


    # Map recipe names to their modules
    recipe_map = {
        'lblcompute': lbl_compute,
        'lbltemplate': lbl_template,
        'lblmask': lbl_mask,
        'lblfind': lbl_find,
        'lblsetup': lbl_setup,
        'lblcompile': lbl_compile,
        'lblnoise': lbl_noise,
        'lblwrap': lbl_wrap,
        'lblreset': lbl_reset,
        'lblresmap': lbl_resmap,
        'lbltelluclean': lbl_telluclean,
        'lbldemo': lbl_demo,
    }

    if recipe_name is None:
        print("Could not identify recipe from log input")
        return


    # Get the recipe module
    if recipe_name not in recipe_map:
        print(f"Recipe '{recipe_name}' not found. Available: {list(recipe_map.keys())}")
        return

    recipe_module = recipe_map[recipe_name]

    print("\n\n")
    print(f"from lbl.recipes import {recipe_module.__NAME__}")
    print(f"{recipe}.main(", end="")
    print(f"{recipe_module.__NAME__}.main(", end="")
    print(", ".join(f"{k}={v}" for k, v in args.items()), end="")
    print("\n\n")
    # ask user whether to run recipe
    user_input = input("Do you want to run (y/n): ")

    if user_input.lower() != 'y':
        # Call main with parsed arguments
        try:
            recipe_module.main(**arguments)
        except Exception as e:
            print(f"Error running {recipe_name}: {e}")
            raise


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # Parse log and print extracted information
    recipe, args = parse_lbl_log(apero_input)
    print("=" * 80)
    print("Parsed Log Information")
    print("=" * 80)
    print(f"Recipe Name: {recipe}")
    print(f"\nExtracted Arguments:")
    for key, value in args.items():
        print(f"  {key}={value}")

    # To run the actual LBL command, uncomment the line below:
    run_lbl_from_log(recipe, args)

# =============================================================================
# End of code
# =============================================================================
