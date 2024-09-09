import pandas as pd
from sqlalchemy import create_engine, text, insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import insert as pg_insert
from utils.settings import POSTGRES_DB_URL

class DBConnection:
    def __init__(self, db_url=POSTGRES_DB_URL):
        """Initialize the DBConnection class with the database URL."""
        self.db_url = db_url
        self.engine = None

    def create_engine(self) -> None:
        """Create the SQLAlchemy engine."""
        if self.engine is None:
            self.engine = create_engine(self.db_url)
        else:
            print("Engine already created.")

    def test_connection(self) -> bool:
        """Test the database connection by executing a simple query."""
        if self.engine is None:
            print("Engine is not created yet.")
            return False

        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            print("Database connection successful!")
            return True
        except SQLAlchemyError as e:
            print(f"Error connecting to the database: {e}")
            return False

    def pull_data_to_dataframe(self, query: str) -> pd.DataFrame:
        """Execute the SQL query and return the result as a Pandas DataFrame."""
        if self.engine is None:
            print("Engine is not created yet.")
            return None
        
        if not query:
            print("Query is empty or None.")
            return None

        try:
            df = pd.read_sql(text(query), self.engine)
            return df
        except SQLAlchemyError as e:
            print(f"Error executing query: {e}")
            return None
    
    def insert_data(self, table_name: str, data: list[dict]) -> bool:
        """Insert data into the specified table."""
        if self.engine is None:
            print("Engine is not created yet.")
            return False
        
        if not data:
            print("Data is empty or None.")
            return False

        try:
            with self.engine.connect() as connection:
                # Generate an insert statement
                insert_stmt = text(f"""
                    INSERT INTO {table_name} ({', '.join(data[0].keys())})
                    VALUES ({', '.join([':' + k for k in data[0].keys()])})
                """)
                # Execute the insert statement for each row of data
                connection.execute(insert_stmt, data)
            print(f"Data successfully inserted into {table_name}.")
            return True
        except SQLAlchemyError as e:
            print(f"Error inserting data: {e}")
            return False



    def upsert_data(self, table_name: str, data: list[dict], conflict_columns: list[str]) -> bool:
        """Perform an upsert operation on the specified table."""
        if self.engine is None:
            print("Engine is not created yet.")
            return False
        
        if not data or not conflict_columns:
            print("Data or conflict columns are empty or None.")
            return False

        try:
            with self.engine.connect() as connection:
                # Prepare the columns and values for insertion
                columns = data[0].keys()
                values = [{col: row[col] for col in columns} for row in data]
                
                # Create an insert statement with on conflict do update
                stmt = pg_insert(table_name).values(values)
                
                # Define the update dictionary (excluding conflict columns)
                update_dict = {col: stmt.excluded[col] for col in columns if col not in conflict_columns}
                
                # Construct the on conflict statement
                on_conflict_stmt = stmt.on_conflict_do_update(
                    index_elements=conflict_columns,
                    set_=update_dict
                )
                
                # Execute the upsert statement
                connection.execute(on_conflict_stmt)
            print(f"Data successfully upserted into {table_name}.")
            return True
        except SQLAlchemyError as e:
            print(f"Error upserting data: {e}")
            return False


    def close(self) -> None:
        """Close the engine and cleanup resources."""
        if self.engine is not None:
            self.engine.dispose()
            print("Database engine disposed.")
        else:
            print("Engine is not created yet.")
