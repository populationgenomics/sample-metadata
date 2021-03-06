from enum import Enum


class ExportType(Enum):
    """
    Wraps up common properties and allows for parameterisation
    of some table exports.
    """

    CSV = 'csv'
    TSV = 'tsv'
    JSON = 'json'

    def get_extension(self):
        """Get extension (including .)"""
        return {
            ExportType.CSV: '.csv',
            ExportType.TSV: '.tsv',
            ExportType.JSON: '.json',
        }[self]

    def get_delimiter(self):
        """Get delimiter (eg: ',' OR '\t')"""
        return {ExportType.CSV: ',', ExportType.TSV: '\t', ExportType.JSON: None}[self]

    def get_mime_type(self):
        """Get MIME type for response"""
        return {
            ExportType.CSV: 'text/csv',
            ExportType.TSV: 'text/tab-separated-values',
            ExportType.JSON: 'application/json',
        }[self]
