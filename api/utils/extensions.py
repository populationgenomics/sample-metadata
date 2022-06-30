import csv
from typing import Optional


EXTENSION_TO_DELIM_MAP = {
    '.csv': ',',
    '.tsv': '\t',
    '.ped': '\t',
}


def guess_delimiter_by_filename(filename: str, raise_exception=True) -> Optional[str]:
    """
    Guess the delimiter from a filename
    """
    filen = filename.lower()

    for extension, delim in EXTENSION_TO_DELIM_MAP.items():
        if filen.endswith(extension):
            return delim

    if raise_exception:
        raise ValueError(
            f'Unable to determine the delimiter from the filename ({filen}), '
            'please explicitly specify a delimiter'
        )

    return None


def guess_delimiter_by_upload_file_obj(
    file, default_delimiter: Optional[str] = None, raise_exception: bool = True
) -> Optional[str]:
    """
    Guess delimiter from uploaded file object, as a convenience, allow specifying
    a default_delimiter, and do transformations based on that (eg: un-escaping backslash)
    """
    if default_delimiter:
        return default_delimiter.replace('\\t', '\t')

    filename_delimiter: str = guess_delimiter_by_filename(
        file.filename, raise_exception=False
    )

    if filename_delimiter:
        return filename_delimiter

    first_line = file.file.readline().decode()
    sniffed_delim = csv.Sniffer().sniff(first_line).delimiter

    # go back to start of file
    file.file.seek(0)

    if sniffed_delim:
        return sniffed_delim

    if raise_exception:
        raise ValueError(
            'Unable to determine the delimiter from uploaded file with name , '
            f'({file.filename}) please explicitly specify a delimiter'
        )

    return None
