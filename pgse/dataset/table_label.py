from typing import Union

import pandas as pd

from pgse.dataset.label_utils import LabelColumns, as_label_columns, missing_columns, to_float_matrix
from pgse.dataset.sample_source import SampleSource
from pgse.log import logger

# A table is either the path of a CSV file or a DataFrame that is already in memory.
TableArg = Union[str, pd.DataFrame]

DEFAULT_LABEL_COLUMN = 'labels'


class TableLabel(SampleSource):
    """Samples held one per row of a table: one column carries the text, the others the labels."""

    inline = True

    def __init__(
            self,
            table: TableArg,
            data_column: str,
            label_columns: LabelColumns = DEFAULT_LABEL_COLUMN
    ) -> None:
        """
        Args:
            table: Path of the CSV file holding one sample per row, or the DataFrame itself.
            data_column: Name of the column holding the sequence of each sample.
            label_columns: Name of the column holding the target value of each sample, or
                the names of several such columns to train one output per column.
        """
        self.table: TableArg = table
        self.data_column: str = data_column
        self.label_columns: list[str] = as_label_columns(label_columns)

        rows = self._read_rows()
        logger.info(
            f'Read {len(rows)} samples from column {data_column!r} of the table, '
            f'labelled by {self.label_columns}'
        )

        super().__init__(
            rows[self.data_column].astype(str).tolist(),
            to_float_matrix(rows, self.label_columns),
            self.label_columns
        )

    def _read_rows(self) -> pd.DataFrame:
        """Read the table and drop the rows whose sequence or any label is empty."""
        data = self.table if isinstance(self.table, pd.DataFrame) else pd.read_csv(self.table)

        missing = missing_columns(data, [self.data_column] + self.label_columns)
        if missing:
            raise ValueError(
                f'The table has no column {missing} to read. Its columns are {list(data.columns)}.'
            )

        text = data[self.data_column]
        labelled = data[self.label_columns].notna().all(axis=1)
        kept = data[text.notna() & (text.astype(str).str.strip() != '') & labelled]

        if len(kept) < len(data):
            logger.warning(
                f'ignored {len(data) - len(kept)} rows with an empty '
                f'{self.data_column!r} or {self.label_columns}'
            )
        if kept.empty:
            raise ValueError('The table holds no usable rows.')

        return kept
