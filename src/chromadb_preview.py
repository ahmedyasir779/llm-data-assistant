import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import pandas as pd


class DataVectorStore:
    """Vector store for semantic search over data"""
    
    def __init__(self, collection_name: str = "data_assistant"):
        """Initialize vector store"""
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Data analysis assistant"}
        )
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"✅ Vector store initialized: {collection_name}")
    
    def add_dataframe_context(self, df: pd.DataFrame, file_name: str = "dataset") -> int:
        """Add dataframe information to vector store"""
        documents = []
        metadatas = []
        ids = []
        
        # Add column information
        for idx, col in enumerate(df.columns):
            col_info = self._get_column_info(df, col)
            documents.append(col_info)
            metadatas.append({
                "type": "column",
                "column_name": col,
                "file_name": file_name
            })
            ids.append(f"{file_name}_col_{idx}")
        
        # Add sample rows
        for idx, row in df.head(5).iterrows():
            row_text = self._row_to_text(row)
            documents.append(row_text)
            metadatas.append({
                "type": "sample_row",
                "row_index": int(idx),
                "file_name": file_name
            })
            ids.append(f"{file_name}_row_{idx}")
        
        # Add statistics
        stats_text = self._get_stats_text(df)
        documents.append(stats_text)
        metadatas.append({"type": "statistics", "file_name": file_name})
        ids.append(f"{file_name}_stats")
        
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        
        return len(documents)
    
    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search for relevant context"""
        results = self.collection.query(query_texts=[query], n_results=n_results)
        
        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0]
        }
    
    def get_relevant_context(self, question: str, n_results: int = 3) -> str:
        """Get relevant context for a question"""
        results = self.search(question, n_results)
        
        context_parts = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            context_parts.append(f"[{meta['type']}] {doc}")
        
        return "\n\n".join(context_parts)
    
    def _get_column_info(self, df: pd.DataFrame, column: str) -> str:
        """Get detailed column information"""
        col_data = df[column]
        
        info_parts = [
            f"Column: {column}",
            f"Type: {col_data.dtype}",
            f"Non-null: {col_data.count()}/{len(df)}"
        ]
        
        if col_data.dtype in ['int64', 'float64']:
            info_parts.extend([
                f"Mean: {col_data.mean():.2f}",
                f"Min: {col_data.min()}, Max: {col_data.max()}"
            ])
        else:
            unique_count = col_data.nunique()
            info_parts.append(f"Unique values: {unique_count}")
            if unique_count < 10:
                info_parts.append(f"Values: {', '.join(map(str, col_data.unique()[:10]))}")
        
        return " | ".join(info_parts)
    
    def _row_to_text(self, row: pd.Series) -> str:
        """Convert row to readable text"""
        row_parts = []
        for col, val in row.items():
            row_parts.append(f"{col}: {val}")
        return " | ".join(row_parts)
    
    def _get_stats_text(self, df: pd.DataFrame) -> str:
        """Get overall statistics as text"""
        stats_parts = [
            f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns",
            f"Columns: {', '.join(df.columns)}"
        ]
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            stats_parts.append(f"Numeric columns: {len(numeric_cols)}")
        
        return " | ".join(stats_parts)
    
    def reset(self) -> None:
        """Clear all data from vector store"""
        self.client.reset()
        self.collection = self.client.get_or_create_collection(name=self.collection.name)
        print("✅ Vector store reset")