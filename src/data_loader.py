import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import json


class DataLoader:
    def __init__(self, filepath: str):
        """
        Initialize data loader
        
        Args:
            filepath: Path to data file (CSV, Excel, JSON)
        """
        self.filepath = Path(filepath)
        self.df: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        self._load_data()
        self._analyze_data()
        
        print(f"✓ Loaded {self.filepath.name}: {len(self.df)} rows × {len(self.df.columns)} columns")
    
    def _load_data(self):
        """Load data based on file extension"""
        ext = self.filepath.suffix.lower()
        
        if ext == '.csv':
            self.df = pd.read_csv(self.filepath)
        elif ext in ['.xlsx', '.xls']:
            self.df = pd.read_excel(self.filepath)
        elif ext == '.json':
            self.df = pd.read_json(self.filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def _analyze_data(self):
        """Analyze dataset and create metadata"""
        self.metadata = {
            'name': self.filepath.stem,
            'filepath': str(self.filepath),
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'column_names': list(self.df.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            'missing_values': self._get_missing_info(),
            'statistics': self._get_statistics(),
            'sample_data': self._get_sample_data()
        }
    
    def _get_missing_info(self) -> Dict[str, Any]:
        """Get information about missing values"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        return {
            col: {
                'count': int(missing[col]),
                'percentage': float(missing_pct[col])
            }
            for col in self.df.columns if missing[col] > 0
        }
    
    def _get_statistics(self) -> Dict[str, Any]:
        """Get statistics for numeric columns"""
        stats = {}
        
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            stats[col] = {
                'mean': float(self.df[col].mean()),
                'median': float(self.df[col].median()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'count': int(self.df[col].count())
            }
        
        return stats
    
    def _get_sample_data(self, n: int = 5) -> List[Dict]:
        """Get sample rows as list of dicts"""
        return self.df.head(n).to_dict('records')
    
    def get_context_summary(self) -> str:
        """
        Generate concise context summary for LLM
        
        Returns:
            Formatted context string
        """
        context = f"""Dataset: {self.metadata['name']}
            Size: {self.metadata['rows']} rows × {self.metadata['columns']} columns

            Columns ({len(self.metadata['column_names'])}):
            {self._format_columns()}

            {self._format_statistics()}

            {self._format_missing_data()}

            Sample Data (first 3 rows):
            {self._format_sample_data(3)}
        """
        return context
    
    def _format_columns(self) -> str:
        """Format column information"""
        lines = []
        for col in self.metadata['column_names']:
            dtype = self.metadata['dtypes'][col]
            lines.append(f"  - {col} ({dtype})")
        return '\n'.join(lines)
    
    def _format_statistics(self) -> str:
        """Format statistics"""
        if not self.metadata['statistics']:
            return "No numeric columns."
        
        lines = ["Statistics:"]
        for col, stats in self.metadata['statistics'].items():
            lines.append(f"  {col}:")
            lines.append(f"    Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}")
            lines.append(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
        
        return '\n'.join(lines)
    
    def _format_missing_data(self) -> str:
        """Format missing data information"""
        if not self.metadata['missing_values']:
            return "No missing values."
        
        lines = ["Missing Values:"]
        for col, info in self.metadata['missing_values'].items():
            lines.append(f"  - {col}: {info['count']} ({info['percentage']:.1f}%)")
        
        return '\n'.join(lines)
    
    def _format_sample_data(self, n: int = 3) -> str:
        """Format sample data"""
        sample = self.df.head(n)
        return sample.to_string(index=False)
    
    def get_column_info(self, column_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a column
        
        Args:
            column_name: Name of the column
            
        Returns:
            Column information dictionary
        """
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found")
        
        col_data = self.df[column_name]
        
        info = {
            'name': column_name,
            'dtype': str(col_data.dtype),
            'count': int(col_data.count()),
            'missing': int(col_data.isnull().sum()),
            'unique': int(col_data.nunique())
        }
        
        # Add statistics for numeric columns
        if col_data.dtype in ['int64', 'float64']:
            info['statistics'] = {
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max())
            }
        
        # Add value counts for categorical
        if col_data.dtype == 'object' or col_data.nunique() < 20:
            value_counts = col_data.value_counts().head(10)
            info['top_values'] = {
                str(k): int(v) for k, v in value_counts.items()
            }
        
        # Add sample values
        info['sample_values'] = [str(v) for v in col_data.dropna().head(5).tolist()]
        
        return info
    
    def query_data(self, query: str) -> pd.DataFrame:
        """
        Query data using pandas query syntax
        
        Args:
            query: Pandas query string
            
        Returns:
            Filtered dataframe
        """
        return self.df.query(query)
    
    def get_value_counts(self, column_name: str, top_n: int = 10) -> Dict[str, int]:
        """
        Get value counts for a column
        
        Args:
            column_name: Name of the column
            top_n: Number of top values to return
            
        Returns:
            Dictionary of value counts
        """
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found")
        
        value_counts = self.df[column_name].value_counts().head(top_n)
        return {str(k): int(v) for k, v in value_counts.items()}
    
    def get_correlation(self, col1: str, col2: str) -> float:
        """
        Get correlation between two numeric columns
        
        Args:
            col1: First column name
            col2: Second column name
            
        Returns:
            Correlation coefficient
        """
        return float(self.df[col1].corr(self.df[col2]))
    
    def save_metadata(self, filepath: str = None):
        """
        Save metadata to JSON
        
        Args:
            filepath: Path to save metadata (default: data_name_metadata.json)
        """
        if filepath is None:
            filepath = f"output/{self.metadata['name']}_metadata.json"
        
        Path(filepath).parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"✓ Metadata saved to {filepath}")



if __name__ == "__main__":
    print("🧪 Testing Data Loader\n")
    
    # Create sample data
    import pandas as pd
    
    sample_data = {
        'product': ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard'],
        'price': [1200, 800, 600, 400, 150],
        'rating': [4.5, 3.8, 4.2, 4.7, 3.5],
        'reviews': [450, 1200, 890, 230, 180],
        'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Accessories']
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('data/sample_products.csv', index=False)
    
    print("=" * 60)
    print("TEST 1: Load Data")
    print("=" * 60)
    
    loader = DataLoader('data/sample_products.csv')
    
    print("\n" + "=" * 60)
    print("TEST 2: Context Summary")
    print("=" * 60)
    print(loader.get_context_summary())
    
    print("\n" + "=" * 60)
    print("TEST 3: Column Info")
    print("=" * 60)
    col_info = loader.get_column_info('price')
    print(json.dumps(col_info, indent=2))
    
    print("\n" + "=" * 60)
    print("TEST 4: Value Counts")
    print("=" * 60)
    counts = loader.get_value_counts('category')
    print(json.dumps(counts, indent=2))
    
    print("\n" + "=" * 60)
    print(" All tests complete!")
    print("=" * 60)