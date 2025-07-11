import pandas as pd
import re

def infer_dtype(series: pd.Series) -> pd.Series:
    """ Essaie de convertir une Series pandas en int ou float si possible. """
    # regex int
    if series.str.fullmatch(r'-?\d+').all():
        return series.astype(int)
    # regex float (avec décimales)
    if series.str.fullmatch(r'-?\d+\.\d+').all():
        return series.astype(float)
    return series  # sinon on garde str

def parse_pipe(file_path: str) -> pd.DataFrame:
    """
    Parse any pipe-separated metadata file with comment-lines en ';' et
    une ligne d'entête commentée (contenant '|').

    - file_path : chemin vers le fichier.
    - Retourne un DataFrame pandas où chaque colonne a été typée si possible.
    """
    col_names = None
    records = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')
            # 1) repérer la ligne d'entête (commentée, contenant '|')
            if col_names is None and line.startswith(';') and line.count('|') >= 2:
                # Ex. ";ID    |READER|MINUTES| SUBSET ... "
                header = line.lstrip(';').strip()
                col_names = [col.strip() for col in header.split('|')]
                n_cols = len(col_names)
                continue

            # 2) lignes de données : ni vides, ni commentaires
            if not line or line.startswith(';'):
                continue

            # 3) split avec maxsplit pour ne pas éclater les champs qui contiennent '|'
            parts = line.split('|', n_cols - 1)
            if len(parts) != n_cols:
                raise ValueError(
                    f"Ligne mal formée (attendu {n_cols} champs, trouvé {len(parts)}):\n  {line}"
                )
            # 4) cleaner chaque champ
            values = [p.strip() for p in parts]
            records.append(dict(zip(col_names, values)))

    if col_names is None:
        raise ValueError("Entête introuvable : pas de ligne commentée contenant '|'")

    # 5) construire le DataFrame
    df = pd.DataFrame.from_records(records, columns=col_names)

    # 6) inférer les types colonne par colonne
    for col in df.columns:
        df[col] = infer_dtype(df[col])

    return df


if __name__ == "__main__":
    metadata_path = 'data\LibriSpeech\CHAPTERS.TXT'
    df = parse_pipe(metadata_path)
    print(df.info())
    print(df.head())
