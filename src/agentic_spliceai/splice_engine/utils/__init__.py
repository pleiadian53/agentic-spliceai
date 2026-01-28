"""Shared utility functions for splice engine.

This package provides common utilities used across base_layer and meta_layer:
- DataFrame operations
- File system utilities
- Display/printing utilities

Exports:
    is_dataframe_empty: Check if DataFrame is empty
    subsample_dataframe: Subsample rows from DataFrame
    read_splice_sites: Read splice sites from file
    print_emphasized: Print emphasized text
    print_with_indent: Print with indentation
"""

from .dataframe import (
    is_dataframe_empty,
    smart_read_csv,
    get_n_unique,
    get_unique_values,
    drop_columns,
    subsample_dataframe,
    align_and_append,
    filter_by_column,
)

from .filesystem import (
    ensure_directory,
    read_splice_sites,
    save_dataframe,
    list_files,
    get_file_size,
    format_file_size,
)

from .display import (
    print_emphasized,
    print_with_indent,
    print_section_separator,
    display,
    display_dataframe_in_chunks,
    format_time,
    format_count,
)

__all__ = [
    # DataFrame utilities
    'is_dataframe_empty',
    'smart_read_csv',
    'get_n_unique',
    'get_unique_values',
    'drop_columns',
    'subsample_dataframe',
    'align_and_append',
    'filter_by_column',
    # Filesystem utilities
    'ensure_directory',
    'read_splice_sites',
    'save_dataframe',
    'list_files',
    'get_file_size',
    'format_file_size',
    # Display utilities
    'print_emphasized',
    'print_with_indent',
    'print_section_separator',
    'display',
    'display_dataframe_in_chunks',
    'format_time',
    'format_count',
]
