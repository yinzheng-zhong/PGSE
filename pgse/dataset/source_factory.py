"""Factory building the sample source a pipeline reads its dataset from."""

from typing import Optional

from pgse.dataset.file_label import FileLabel
from pgse.dataset.label_utils import LabelColumns
from pgse.dataset.sample_source import SampleSource
from pgse.dataset.table_label import DEFAULT_LABEL_COLUMN, TableArg, TableLabel


def build_source(
        data_dir: Optional[str] = None,
        label_file: Optional[str | dict] = None,
        pre_kfold_info_file: Optional[str] = None,
        table_file: Optional[TableArg] = None,
        data_column: Optional[str] = None,
        label_columns: Optional[LabelColumns] = None
) -> SampleSource:
    """Build a table source when a table is given, and a file source otherwise.

    Args:
        data_dir: Directory holding the sequence files, read in file mode.
        label_file: CSV file, or dict, pairing each file with its labels, read in file mode.
        pre_kfold_info_file: JSON file holding predefined folds, read in file mode.
        table_file: CSV file, or DataFrame, holding one sample per row, read in table mode.
        data_column: Name of the column holding the sequences, required in table mode.
        label_columns: Name of the column holding the labels, or the names of several such
            columns. Required in file mode, and defaults to 'labels' in table mode.
    """
    if table_file is None:
        if label_file is None:
            raise ValueError('Pass either label_file with data_dir, or table_file with data_column.')
        return FileLabel(label_file, data_dir, pre_kfold_info_file, label_columns)

    if not data_column:
        raise ValueError('table_file needs data_column, the name of the column holding the sequences.')

    return TableLabel(table_file, data_column, label_columns or DEFAULT_LABEL_COLUMN)
