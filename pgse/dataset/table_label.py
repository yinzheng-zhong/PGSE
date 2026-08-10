from typing import Union

import pandas as pd

from pgse.dataset.label_utils import to_float_labels
from pgse.dataset.sample_source import SampleSource
from pgse.log import logger

# A table is either the path of a CSV file or a DataFrame that is already in memory.
TableArg = Union[str, pd.DataFrame]

DEFAULT_LABEL_COLUMN = 'labels'


class TableLabel(SampleSource):
    """Samples held one per row of a table: one column carries the text, another the label."""

    inline = True

    def __init__(
            self,
            table: TableArg,
            data_column: str,
            label_column: str = DEFAULT_LABEL_COLUMN
    ) -> None:
        """
        Args:
            table: Path of the CSV file holding one sample per row, or the DataFrame itself.
            data_column: Name of the column holding the sequence of each sample.
            label_column: Name of the column holding the target value of each sample.
        """
        self.table: TableArg = table
        self.data_column: str = data_column
        self.label_column: str = label_column

        rows = self._read_rows()
        logger.info(f'Read {len(rows)} samples from column {data_column!r} of the table')

        super().__init__(
            rows[data_column].astype(str).tolist(),
            to_float_labels(rows[label_column], label_column)
        )

    def _read_rows(self) -> pd.DataFrame:
        """Read the table and drop the rows whose sequence or label is empty."""
        data = self.table if isinstance(self.table, pd.DataFrame) else pd.read_csv(self.table)

        missing = [column for column in (self.data_column, self.label_column) if column not in data.columns]
        if missing:
            raise ValueError(
                f'The table has no column {missing} to read. Its columns are {list(data.columns)}.'
            )

        text = data[self.data_column]
        kept = data[text.notna() & (text.astype(str).str.strip() != '') & data[self.label_column].notna()]

        if len(kept) < len(data):
            logger.warning(
                f'ignored {len(data) - len(kept)} rows with an empty '
                f'{self.data_column!r} or {self.label_column!r}'
            )
        if kept.empty:
            raise ValueError('The table holds no usable rows.')

        return kept
