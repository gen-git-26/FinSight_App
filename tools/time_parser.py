# tools/time_parser.py
"""
Smart time range parser - converts natural language time expressions to days/dates.
Handles: "30 days", "15 weeks", "3 months", "1 year", "6 months", etc.
"""

import re
from typing import Optional, Tuple
from datetime import datetime, timedelta


def parse_time_range_to_days(time_str: Optional[str]) -> Optional[int]:
    """
    Convert time range string to number of days.
    
    Examples:
        "30 days" → 30
        "2 weeks" → 14
        "3 months" → 90
        "1 year" → 365
        "15 weeks" → 105
        "6 months" → 180
    
    Args:
        time_str: Time range string (can be None)
    
    Returns:
        Number of days, or None if parsing failed
    """
    if not time_str:
        return None
    
    s = time_str.lower().strip()
    
    # Try multiple regex patterns
    patterns = [
        r'(\d+(?:\.\d+)?)\s*(days?|day)',      # "30 days"
        r'(\d+(?:\.\d+)?)\s*(weeks?|week)',    # "2 weeks"
        r'(\d+(?:\.\d+)?)\s*(months?|month)',  # "3 months"
        r'(\d+(?:\.\d+)?)\s*(years?|year)',    # "1 year"
        r'(\d+(?:\.\d+)?)\s*([dwmy])',         # "30d", "2w", "3m", "1y" (shorthand)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, s)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            
            # Normalize unit to first letter
            unit_map = {
                'd': 'd', 'day': 'd', 'days': 'd',
                'w': 'w', 'week': 'w', 'weeks': 'w',
                'm': 'm', 'month': 'm', 'months': 'm',
                'y': 'y', 'year': 'y', 'years': 'y',
            }
            
            unit_normalized = unit_map.get(unit, None)
            if not unit_normalized:
                continue
            
            # Convert to days
            conversions = {
                'd': 1,           # 1 day = 1 day
                'w': 7,           # 1 week = 7 days
                'm': 30,          # 1 month ≈ 30 days (conservative)
                'y': 365          # 1 year ≈ 365 days
            }
            
            days = int(value * conversions.get(unit_normalized, 1))
            
            # Sanity check: days should be positive and reasonable
            if 0 < days <= 3650:  # max 10 years
                return days
    
    return None


def parse_time_range_to_date(
    time_str: Optional[str],
    from_date: Optional[datetime] = None,
    format_str: str = "%Y-%m-%d"
) -> Optional[str]:
    """
    Convert relative time range to absolute date.
    
    Examples:
        "30 days" → "2025-11-17"  (from today)
        "15 weeks" → "2026-01-05"
        "3 months" → "2026-01-18"
    
    Args:
        time_str: Time range string
        from_date: Base date (defaults to today)
        format_str: Output date format
    
    Returns:
        Formatted date string, or None if parsing failed
    """
    if not from_date:
        from_date = datetime.now()
    
    days = parse_time_range_to_days(time_str)
    
    if days is None:
        return None
    
    target_date = from_date + timedelta(days=days)
    return target_date.strftime(format_str)


def extract_time_range_and_unit(
    time_str: Optional[str]
) -> Optional[Tuple[int, str]]:
    """
    Extract numeric value and unit from time string.
    
    Examples:
        "30 days" → (30, "days")
        "15 weeks" → (15, "weeks")
        "3m" → (3, "months")
    
    Args:
        time_str: Time range string
    
    Returns:
        Tuple of (value, unit_name), or None if parsing failed
    """
    if not time_str:
        return None
    
    s = time_str.lower().strip()
    
    pattern = r'(\d+(?:\.\d+)?)\s*([dwmy]|\w+)?'
    match = re.search(pattern, s)
    
    if not match:
        return None
    
    value = int(float(match.group(1)))
    unit = match.group(2) or "days"
    
    unit_map = {
        'd': 'days',
        'w': 'weeks',
        'm': 'months',
        'y': 'years',
        'day': 'days',
        'week': 'weeks',
        'month': 'months',
        'year': 'years',
    }
    
    unit_normalized = unit_map.get(unit, unit)
    
    return (value, unit_normalized)


def normalize_time_expression(time_str: Optional[str]) -> Optional[str]:
    """
    Normalize time expression to standard format.
    
    Examples:
        "30d" → "30 days"
        "2w" → "2 weeks"
        "3m" → "3 months"
        "1y" → "1 year"
    
    Args:
        time_str: Time range string
    
    Returns:
        Normalized string, or None if couldn't parse
    """
    extracted = extract_time_range_and_unit(time_str)
    if not extracted:
        return None
    
    value, unit = extracted
    return f"{value} {unit}"

