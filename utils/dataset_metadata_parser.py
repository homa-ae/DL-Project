import pandas as pd

def infer_dtype(series: pd.Series) -> pd.Series:
    """Convert a pandas.Series to int or float when possible.

    Args:
        series (pd.Series): raw column

    Returns:
        pd.Series: formated column
    """
    # regex int
    if series.str.fullmatch(r'-?\d+').all():
        return series.astype(int)
    # regex float (avec décimales)
    if series.str.fullmatch(r'-?\d+\.\d+').all():
        return series.astype(float)
    return series  # sinon on garde str

def parse_pipe(file_path: str) -> pd.DataFrame:
    """Parse pipe-separated (|) metadata files into a formated pandas.DataFrame.

    Args:
        file_path (str): Input raw LibriVox metadata file

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        pd.DataFrame: formated metadata pandas.DataFrame
    """
    col_names = None
    records = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')

            # Detect the header line (commented, containing at least two '|')
            if col_names is None and line.startswith(';') and line.count('|') >= 2:
                # e.g. ";ID    |READER|MINUTES| SUBSET ... "
                header = line.lstrip(';').strip()
                col_names = [col.strip() for col in header.split('|')]
                n_cols = len(col_names)
                continue

            # Skip empty lines or comments (;)
            if not line or line.startswith(';'):
                continue

            # Split with maxsplit to avoid splitting fields that contain '|' (e.g. " |CBW|Simon")
            parts = line.split('|', n_cols - 1)
            if len(parts) != n_cols:
                raise ValueError(
                    f"Malformed line (expected {n_cols} fields, found {len(parts)}):\n  {line}"
                )
            # Clean each field
            values = [p.strip() for p in parts]
            records.append(dict(zip(col_names, values)))

    if col_names is None:
        raise ValueError("Header not found: no commented line containing \'|\'")

    # Create the DataFrame
    df = pd.DataFrame.from_records(records, columns=col_names)

    # Infer column types one by one
    for col in df.columns:
        df[col] = infer_dtype(df[col])

    return df


if __name__ == "__main__":
    metadata_path = 'data\LibriSpeech\CHAPTERS.TXT'
    df = parse_pipe(metadata_path)
    print(df.info())
    print(df.head())
