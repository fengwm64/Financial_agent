import sqlite3
from typing import List, Tuple
import logging
from utils.config import Config

class DB:
    def __init__(self, p_db_path: str) -> None:
        try:
            self.conn = sqlite3.connect(p_db_path)
        except sqlite3.Error as e:
            logging.error(f"Failed to connect to the database: {e}")
            raise RuntimeError(f"Failed to connect to the database: {e}")

    def __del__(self):
        self.close()

    def close(self) -> None:
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            logging.debug("Database connection closed.")

    def commit(self) -> None:
        try:
            self.conn.commit()
            logging.debug("Transaction committed.")
        except sqlite3.Error as e:
            logging.error(f"Failed to commit the transaction: {e}")
            raise RuntimeError(f"Failed to commit the transaction: {e}")

    def select(self, p_sql_str: str) -> List[Tuple]:
        try:
            # 创建游标
            cur = self.conn.cursor()
            cur.execute(p_sql_str)
            sql_answer = cur.fetchall()
            cur.close()  # 确保游标被关闭
            
            if len(sql_answer) > 10:
                logging.warning("Query returned more than 10 results, which is too many.")
                raise ValueError("Query returned more than 10 results, which is too many.")
            
            logging.debug(f"Executed query: {p_sql_str}")
            return sql_answer
        
        except sqlite3.OperationalError as e:
            logging.error(f"Operational error: {e} with query: {p_sql_str}")
            raise ValueError(f"SQL query failed due to operational error: {e}")
        
        except sqlite3.ProgrammingError as e:
            logging.error(f"Programming error: {e} with query: {p_sql_str}")
            raise ValueError(f"SQL query failed due to programming error: {e}")
        
        except sqlite3.IntegrityError as e:
            logging.error(f"Integrity error: {e} with query: {p_sql_str}")
            raise ValueError(f"SQL query failed due to integrity error: {e}")
        
        except sqlite3.InterfaceError as e:
            logging.error(f"Interface error: {e} with query: {p_sql_str}")
            raise ValueError(f"SQL query failed due to interface error: {e}")
        
        except sqlite3.DatabaseError as e:
            logging.error(f"Database error: {e} with query: {p_sql_str}")
            raise ValueError(f"Database error occurred: {e}")
        
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e} with query: {p_sql_str}")
            raise RuntimeError(f"An unexpected error occurred: {e}")