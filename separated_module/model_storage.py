"""ModelStorage: persistence of models, weights and memories (SQLite / Rust backend)."""

from datetime import datetime
from typing import Any
import ast
import base64
import io
import joblib
import json
import logging
import numpy as np
import os
import re
import sqlite3
import sys
import traceback
from .common import APP_BASE_DIR, _RUST_MODULE_AVAILABLE, wc

logger = logging.getLogger(__name__)


# Model storage module that handles saving and loading of trained models, their versions, and associated metadata to a database for persistence and future retrieval.
class ModelStorage:
    """SQLite persistence layer for models, weights and memories.

    Creates and manages the tables (models, weights, nodes, agents, HNSW, per-head memories, answer
    cache, explainability log) and saves/loads MLP, Transformer, LSTM, peer and pipeline state. Uses the
    Rust weight store (`abstract_weights_core`) when available and plain sqlite3 otherwise.
    """
    def __init__(self, pipeline, memory_name, db_path='activity_log.db'):
        self.pipeline = pipeline
        self.db_path = db_path

        self.setup_storage_table()
        self.setup_explainable_table()
        self.setup_agent_table()
        self.setup_node_table()
        self.setup_weight_table()
        self.setup_accurate_cache_table()
        self.setup_hnsw_table()
        self.setup_head_memory_table()

        self.memory_name = memory_name

        if not self.memory_exists(self.memory_name, type='Peer'):
            self.id_history = []
        else:
            print(f'|| Found Matched ID from memory: {self.memory_name}!')
            self.id_history = self.load_agent_id(self.memory_name)

    def get_database_path(self):
        db_filename= self.db_path
        if getattr(sys, 'frozen', False):
            application_path = sys._MEIPASS
            print(f"[🔄] Running as EXE, temp path: {application_path}")
        else:
            application_path = APP_BASE_DIR
            print(f"[🔄] Running as script, path: {application_path}")
    
        db_path = os.path.join(application_path, db_filename)
        print(f"[🔄] Looking for database at: {db_path}")
        print(f"[✅] Database exists: {os.path.exists(db_path)}")
    
        return db_path

    def setup_explainable_table(self):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS model_attn_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      model_data TEXT,
                      is_active INTEGER DEFAULT 0,                      
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()
            print('|| Update Attention Saved to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS model_attn_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      model_data TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()


    def setup_storage_table(self):
        """Create the main model/memory storage table if it does not exist."""
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS model_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_version TEXT,
                      model_type TEXT,
                      model_data TEXT,  -- JSON string for dict
                      model_binary BLOB,  -- For pickle files
                      trained_on TEXT,
                      metadata TEXT,  -- JSON for extra info
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()
            print('|| Update Saved to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS model_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_version TEXT,
                      model_type TEXT,
                      model_data TEXT,  -- JSON string for dict
                      model_binary BLOB,  -- For pickle files
                      trained_on TEXT,
                      metadata TEXT,  -- JSON for extra info
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()


    def get_database_path(self):
        """Resolve the database file path (next to the EXE / project folder). Defined twice in the class; the later definition is in effect."""
        db_filename = self.db_path
        print('[>] Getting Database path (activity_log.db)...')
        if getattr(sys, 'frozen', False):
            application_path = sys._MEIPASS
            print(f"[DATABASE] Running as EXE, temp path: {application_path}")
        else:
            application_path = APP_BASE_DIR
            print(f"[DATABASE] Running as script, path: {application_path}")
    
        db_path = os.path.join(application_path, db_filename)
        print(f"[DATABASE] Looking for database at: {db_path}")
        print(f"[DATABASE] Database exists: {os.path.exists(db_path)}")
    
        return db_path


    def setup_explainable_table(self):
        """Create the explainability log table if it does not exist. Defined twice in the class; the later definition is in effect."""
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS model_attn_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      model_data TEXT,
                      is_active INTEGER DEFAULT 0,                      
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()
            print('|| Update Attention Saved to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS model_attn_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      model_data TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()

    def setup_node_table(self):
        """Create the table for query-node memories if it does not exist."""
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS node_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      node_data TEXT,
                      node_id TEXT,
                      is_active INTEGER DEFAULT 0,                      
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()
            print('|| Update Node Saved to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS node_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      node_data TEXT,
                      node_id TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()


    def setup_agent_table(self):
        """Create the table for agent ids / peer data if it does not exist."""
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS agent_attn_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      model_attn_data TEXT,
                      model_target_pred TEXT,
                      agent_id TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''') 
        
            conn.commit()
            conn.close()
            print('|| Update Agent Saved to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS agent_attn_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      model_attn_data TEXT,
                      model_target_pred TEXT,
                      agent_id TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
        
            conn.commit()
            conn.close()


    def setup_weight_table(self):
        """Create the table for serialised model weights if it does not exist."""
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS weight_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      weights TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''') 
        
            conn.commit()
            conn.close()
            print('|| Update Agent Saved to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS weight_storage
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      weights TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''') 
        
        
            conn.commit()
            conn.close()
            
    def setup_hnsw_table(self):
        """Create the table for serialised HNSW indices if it does not exist."""
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS hnsw_weight_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      weights TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''') 
        
            conn.commit()
            conn.close()
            print('|| Update Agent Saved to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS hnsw_weight_storage
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      weights TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''') 
        
        
            conn.commit()
            conn.close()
            
    def setup_head_memory_table(self):
        """Create the table for per-head attention memories if it does not exist."""
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS head_memory_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      weights TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''') 
        
            conn.commit()
            conn.close()
            print('|| Update Agent Saved to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS head_memory_storage
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      weights TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''') 
        
        
            conn.commit()
            conn.close()



    def setup_accurate_cache_table(self):
        """Create the table for the verified-answer cache if it does not exist."""
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS accurate_cache_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      cache TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''') 
        
            conn.commit()
            conn.close()
            print('|| Update cached to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS accurate_cache_storage
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      cache TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''') 
        
        
            conn.commit()
            conn.close() 

    def save_accurate_cache_dict(self, memory_name, payload, model_type='Pipeline'):
        """Persist the verified-answer cache payload for `memory_name`."""
        try:
            db_path = self.get_database_path()
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()

        cache = json.dumps(payload, default=str)

        try:
            c.execute("""
                INSERT INTO accurate_cache_storage
                (memory_name, cache, is_active)
                VALUES (?, ?, ?)
            """, (memory_name, cache, 1))
        
            c.execute("""
                UPDATE accurate_cache_storage
                SET is_active = 0 
                WHERE memory_name = ? AND id != last_insert_rowid()
            """, (memory_name,)) 

            conn.commit()
            conn.close()

            print('|| Accurate cache saved!')

        except Exception as e:
            print(f'[-] Cant save accurate cache memory due to: {e}') 
            pass      

   

    def save_model_dict(self, memory_name, model_dict, type=None, model_type='mlp'):
        """Serialise a model dict (JSON/arrays) and store it under `memory_name` for the given model type."""
        try:
            db_path = self.get_database_path()            
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()

        model_dict = self.pipeline._sanitize_for_storage(model_dict)  
        serializable_dict = self._prepare_for_serialization(model_dict)
        model_json = json.dumps(serializable_dict, default=str)

        if _RUST_MODULE_AVAILABLE:
            try:
                wc.save_pipelines_dict(self.db_path, memory_name, 
                                    model_type, model_json)
                print('[=] Pipelines dictionary saved using Rust module!')
            except Exception as e:
                print(f'[!] Cant save Pipelines dictionary due to: {e}')
                
        else:
            if type == 'Transformer':
                try:
                    c.execute("""
                        INSERT INTO model_attn_storage 
                        (memory_name, model_type, model_data, is_active)
                        VALUES (?, ?, ?, ?)
                    """, (memory_name, model_type, model_json, 1))
            
                    c.execute("""
                        UPDATE model_attn_storage 
                        SET is_active = 0 
                        WHERE memory_name = ? AND id != last_insert_rowid()
                    """, (memory_name,))

                except Exception as e:
                    print(f'[-] Cant save model memory due to: {e}') 
                    pass             
            else:
                try:
                    c.execute("""
                        INSERT INTO model_storage
                        (memory_name, model_type, model_data, is_active)
                        VALUES (?, ?, ?, ?)
                    """, (memory_name, model_type, model_json, 1))
            
                    c.execute("""
                        UPDATE model_storage 
                        SET is_active = 0 
                        WHERE memory_name = ? AND id != last_insert_rowid()
                    """, (memory_name,)) 

                except Exception as e:
                    print(f'[-] Cant save model memory due to: {e}') 
                    pass          
            
        conn.commit()
        model_id = c.lastrowid        
        conn.close()
        
        print(f"✅ Memory '{memory_name}' saved as dict (ID: {model_id})")
        return model_id

    def _prepare_for_serialization(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: self._prepare_for_serialization(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._prepare_for_serialization(item) for item in obj]
        return obj
        


    def load_model_dict(self, memory_name):
        """Load the stored model dict for `memory_name`."""
        conn = None
        if _RUST_MODULE_AVAILABLE:
            try:
                num_classes = self.pipeline._get_num_classes() or 0
                cleaned_json = wc.load_and_validate_model_dict(
                    self.db_path, memory_name, num_classes
                )
                if cleaned_json is None:
                    return None
                data = json.loads(cleaned_json)
                # the heavy JSON parse + basic schema check already done in Rust
                validation = self._validate_and_repair(data, memory_name)
                return validation
            except Exception as e:
                print(f'[!] Rust load_model_dict failed: {e}')
        else:
            try:
                try:
                    conn = sqlite3.connect(self.db_path)
                except:
                    conn = sqlite3.connect(self.get_database_path())

                c = conn.cursor()
                c.execute("""
                    SELECT model_data FROM model_storage
                    WHERE memory_name = ? AND is_active = 1
                    ORDER BY id DESC LIMIT 1
                """, (memory_name,))

                result = c.fetchone()
                if not result:
                    return None

                data = json.loads(result[0])
                data = self._validate_and_repair(data, memory_name)
                return data   # actually return data

            except Exception as e:
                print(f'[!] Error loading model dict: {e}')
                return None
            finally:
                if conn:
                    conn.close()


    def _validate_and_repair(self, data, memory_name=None):
        """Validate loaded data and repair if corrupted."""

        if data is None:
            return {}

        # get num_classes safely without touching pipeline
        num_classes = 0
        try:
            # try pipeline first
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                num_classes = self.pipeline._get_num_classes()
            # fallback — infer from data itself
            elif isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, (list, np.ndarray)):
                        arr = np.asarray(v)
                        if arr.ndim == 1:
                            num_classes = len(arr)
                            break
        except Exception as e:
            print(f'[!] Could not determine num_classes: {e}')
            num_classes = 0

        # handle list data
        if isinstance(data, list) and len(data) > 0:
            if num_classes > 0 and len(data) != num_classes:
                print(f'[!] Shape mismatch: got {len(data)}, expected {num_classes} — repairing')
                return {}

            if all(isinstance(x, (int, float)) for x in data[:10]):
                print(f'[!] List appears to be probabilities, wrapping')
                return {'_cached_probs': np.array(data, dtype=np.float64)}

        # handle dict data
        if isinstance(data, dict):
            repaired = {}
            for key, value in data.items():

                # dynamic corruption check 
                if isinstance(value, list):
                    arr = np.asarray(value)
                    if arr.ndim > 2:
                        print(f'[!] Corrupted value for key {key} '
                            f'(ndim={arr.ndim}), removing')
                        continue

                    if num_classes > 0 and arr.ndim == 1 and \
                    len(arr) not in (num_classes, num_classes * 2):
                        print(f'[!] Suspicious shape for key {key}: '
                            f'{arr.shape}, expected {num_classes} — removing')
                        continue

                # None values — skipped silently
                if value is None:
                    continue

                repaired[key] = value

            return repaired

        return data


    def _convert_to_arrays(self, data):
        """
        Recursively convert data to numpy arrays where possible.
        Safe for ARM64 and handles all data types.
        """
        if data is None:
            return None
        
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                converted = self._convert_value(value)
                if converted is not None:
                    result[key] = converted
            return result
        
        elif isinstance(data, (list, tuple)):
            return [self._convert_value(item) for item in data]
        
        else:
            return self._convert_value(data)



    def _convert_value(self, value):
        """
        Convert a single value to appropriate type.
        Returns original value if conversion fails.
        """
        if value is None:
            return None
        
        # Already numpy array - keep as is
        if isinstance(value, np.ndarray):
            return value
        
        # Handle lists recursively
        if isinstance(value, (list, tuple)):
            return [self._convert_value(item) for item in value]
        
        # Handle dicts recursively
        if isinstance(value, dict):
            return self._convert_to_arrays(value)
        
        # Handle string that might represent an array
        if isinstance(value, str):
            return self._parse_array_string(value)
        
        # Return as-is for other types (int, float, bool, etc.)
        return value

    def save_target_memory_row(self, memory_name, blob):
        """Store the serialised TargetMemory blob for `memory_name`."""
        db_path = self.db_path
        conn = sqlite3.connect(db_path)
        try:
            c = conn.cursor()
            c.execute("INSERT INTO weight_storage (memory_name, model_type, weights, is_active) VALUES (?,?,?,1)",
                    (memory_name, "target_memory", sqlite3.Binary(blob)))
            c.execute("UPDATE weight_storage SET is_active = 0 WHERE memory_name = ? AND model_type = 'target_memory' "
                    "AND id != last_insert_rowid()", (memory_name,))
            conn.commit()
        finally:
            conn.close()
    
    
    def load_target_memory_row(self, memory_name):
        """Load the serialised TargetMemory blob for `memory_name` (None if absent)."""
        db_path = self.db_path
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute("SELECT weights FROM weight_storage WHERE memory_name = ? AND model_type = 'target_memory' "
                            "AND is_active = 1 ORDER BY id DESC LIMIT 1", (memory_name,)).fetchone()
        finally:
            conn.close()
        return bytes(row[0]) if row else None
 

    def _parse_array_string(self, s):
        """
        Parse string representation of array back to numpy array.
        Returns original string if parsing fails.
        """
        if not isinstance(s, str) or not s:
            return s

        if _RUST_MODULE_AVAILABLE:
            try:
                data = wc.parse_array_string(self.db_path, s)
                print('[+] Data successfully parsed!')
                return data
            except Exception as e:
                print(f'[!] Data cant be parsed due to: {e}')
                s = s.replace('\n', '').replace('\r', '').replace('\t', '')
                s = ' '.join(s.split()).strip()
                
                if not s:
                    return s
                
                # parsing as JSON array first
                if s.startswith('[') and s.endswith(']'):
                    try:
                        parsed = json.loads(s)
                        if isinstance(parsed, list):
                            return np.array(parsed, dtype=np.float32)
                    except (json.JSONDecodeError, ValueError):
                        pass
                    
                    # Try parsing with ast.literal_eval
                    try:
                        parsed = ast.literal_eval(s)
                        if isinstance(parsed, (list, tuple)):
                            return np.array(parsed, dtype=np.float32)
                    except (ValueError, SyntaxError, TypeError):
                        pass
                
                # parsing space-separated numbers
                if re.fullmatch(r'[\[\]\s\d\.\,\-\+E]+', s):        
                    parts = s.replace('[', ' ').replace(']', ' ').split()
                    if parts:
                        try:
                            float_values = [float(x) for x in parts]
                            return np.array(float_values, dtype=np.float32)
                        except ValueError:
                            pass
                
                # Handle comma-separated values
                if ',' in s:
                    cleaned = s.replace('[', '').replace(']', '').strip()
                    parts = [p.strip() for p in cleaned.split(',') if p.strip()]
                    try:
                        float_values = [float(x) for x in parts]
                        return np.array(float_values, dtype=np.float32)
                    except ValueError:
                        pass  

                return s         
        else:
            # Clean the string
            s = s.replace('\n', '').replace('\r', '').replace('\t', '')
            s = ' '.join(s.split()).strip()
            
            if not s:
                return s
            
            # parsing as JSON array first
            if s.startswith('[') and s.endswith(']'):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        return np.array(parsed, dtype=np.float32)
                except (json.JSONDecodeError, ValueError):
                    pass
                
                # Try parsing with ast.literal_eval
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple)):
                        return np.array(parsed, dtype=np.float32)
                except (ValueError, SyntaxError, TypeError):
                    pass
            
            # parsing space-separated numbers
            if re.fullmatch(r'[\[\]\s\d\.\,\-\+E]+', s):        
                parts = s.replace('[', ' ').replace(']', ' ').split()
                if parts:
                    try:
                        float_values = [float(x) for x in parts]
                        return np.array(float_values, dtype=np.float32)
                    except ValueError:
                        pass
            
            # Handle comma-separated values
            if ',' in s:
                cleaned = s.replace('[', '').replace(']', '').strip()
                parts = [p.strip() for p in cleaned.split(',') if p.strip()]
                try:
                    float_values = [float(x) for x in parts]
                    return np.array(float_values, dtype=np.float32)
                except ValueError:
                    pass
        
        # Return original string if nothing worked
        return s


    def _convertables_utility(self, memory_name, data, data2, type_func=None, verbose=False):
        """
        Convert and display memory data safely.
        Returns tuple (result, result2) always for consistent return type.
        """
        name = memory_name
        
        # Initialize results
        result = None
        result2 = None
        
        # Convert data based on type_func
        if type_func == "TwoPass" and data2 is not None:
            print('|| Two pass utility converting.')
            result = self._convert_to_arrays(data)
            result2 = self._convert_to_arrays(data2)
        else:
            result = self._convert_to_arrays(data)
        
        # Verify result is a dictionary before calling .items()
        if verbose and result is not None:
            print(f"[=] Retrieved memory: {name}")
            
            # ✅ SAFE: Check if result is a dict before iterating
            if isinstance(result, dict):
                for key, value in result.items():
                    self._print_memory_value(key, value)
            else:
                print(f"[!] Result is not a dict: {type(result)}")
                print(f"[!] Result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
        
        # Handle TwoPass verbose output
        if verbose and data2 is not None and result2 is not None:
            print(f"[=] Retrieved secondary memory: {name}_secondary")
            if isinstance(result2, dict):
                for key, value in result2.items():
                    self._print_memory_value(key, value)
            else:
                print(f"[!] Secondary result is not a dict: {type(result2)}")
        
        # ✅ ALWAYS return consistent types
        if data2 is not None:
            return result, result2
        else:
            return result



    def _print_memory_value(self, key, value):
        # Helper method to print memory values safely
        if isinstance(value, list):
            print(f"  {key}: list of {len(value)} items")
            for i, v in enumerate(value[:5]):  # Limit to first 5 items
                if isinstance(v, np.ndarray):
                    print(f"    [{i}]: array shape {v.shape}")
                else:
                    print(f"    [{i}]: {type(v)}")
            if len(value) > 5:
                print(f"    ... and {len(value) - 5} more items")
        
        elif isinstance(value, np.ndarray):
            print(f"  {key}: array shape {value.shape}, dtype={value.dtype}")
        
        elif isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} keys")
        
        else:
            print(f"  {key}: {type(value)}")


    def memory_retrieval(self, memory_name=None, type_func=None, verbose=False):  
        """Load a saved memory by name; `type_func` ('Transformer', 'Peer', 'Pipeline', ...) selects the loader."""
        name = memory_name

        if type_func == 'Transformer':
            data = self.load_transformer_dict(name)
        elif type_func == 'Peer':
            id_history = self.id_history
          
            first_data, second_data = self.load_peer_request_dict(name, id_history) 
            result, result2 = self._convertables_utility(name, first_data, second_data, type_func='TwoPass', verbose=verbose)
            return result, result2
        elif type_func == 'Node':
            data = self._load_node_dict(name)            
        else:
           data = self.load_model_dict(memory_name)

        if data is None:
            print(f"[-] No memory found: {name}")
            return {}

        result = self._convertables_utility(name, data, None, type_func=type_func, verbose=verbose)

        return result


    def load_accurate_cache(self, memory_name):
        """Load the verified-answer cache for `memory_name`."""
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()      
            
            c.execute("""
            SELECT weights FROM accurate_cache_storage 
            WHERE memory_name = ? AND is_active = 1
            """, (memory_name,))               
        
            result = c.fetchone()
            conn.close()
        
            if result:
                return json.loads(result[0])
        except Exception as e:
            print(f'[!] Error handling cache dict: {e}')
        return None


    def _load_weights(self, memory_name, type=None):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
 
            c.execute("""SELECT weights FROM weight_storage
             WHERE memory_name = ? AND is_active = 1
             AND COALESCE(model_type,'') NOT IN ('mlp','transformer')
             ORDER BY id DESC LIMIT 1""", (memory_name,))              
        
            result = c.fetchone()
            conn.close()
        
            if result:
                return json.loads(result[0])
        except Exception as e:
            print(f'[!] Error handling Weight dict: {e}')
        return None


    def weight_retrieval(self, memory_name=None, type=None, verbose=False):  
        """Load saved weights for `memory_name` (optionally of a given type); None if nothing is stored."""
        name = memory_name

        data = self._load_weights(memory_name, type=type)
        if data is None:
            print(f"[-] No Saved Weight found: {name}")
            return None

        result = self._convertables_utility(name, data, None, type_func='firstpass', verbose=verbose)

        return result       

    def _load_node_dict(self, memory_name):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
        
            c.execute("""
            SELECT node_data FROM node_storage 
            WHERE memory_name = ? AND is_active = 1
            """, (memory_name,))
        
            result = c.fetchone()
            conn.close()
        
            if result:
                return json.loads(result[0])
        except Exception as e:
            print(f'[!] Error handling node dict: {e}')
        return None

    def save_nodes_dict(self, memory_name, node_memory, node_id, model_type='Node'):
        """Persist a query node's memory under `memory_name` / `node_id`."""
        try:
            db_path = self.get_database_path()
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()

        node_json = json.dumps(node_memory, default=str)

        try:
            c.execute("""
                INSERT INTO node_storage 
                (memory_name, model_type, node_data, node_id, is_active)
                VALUES (?, ?, ?, ?, ?)
            """, (memory_name, model_type, node_json, node_id, 1))
        
            c.execute("""
                UPDATE node_storage 
                SET is_active = 0 
                WHERE memory_name = ? AND id != last_insert_rowid()
            """, (memory_name,)) 

            conn.commit()
            conn.close()

            print('[||] Node data dictionary saved!')

        except Exception as e:
            print(f'[-] Cant save Node memory due to: {e}')
            pass  

    def save_hnsw_memory(self, memory_name, model_type='HNSW'):
        """Persist the pipeline's HNSW index memory under `memory_name`."""
        try:
            db_path = self.get_database_path()
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()    

        memories = self.pipeline.model2.memory.hnsw_arch.to_dict()    
        memories_json = json.dumps(memories, default=str)
        try:
            c.execute("""
                INSERT INTO hnsw_weight_storage 
                (memory_name, model_type, weights, is_active)
                VALUES (?, ?, ?, ?)
            """, (memory_name, model_type, memories_json, 1))
        
            c.execute("""
                UPDATE hnsw_weight_storage 
                SET is_active = 0 
                WHERE memory_name = ? AND id != last_insert_rowid()
            """, (memory_name,)) 
            
            conn.commit()
            conn.close()

            print('[||] All HNSW Setup dictionary saved!')

        except Exception as e:
            print(f'[-] Cant save Weights due to: {e}')
            pass 

    def save_memory_head(self, memory_name, model_type='memory_head'):
        """Persist the per-head attention memory under `memory_name`."""
        try:
            db_path = self.get_database_path()
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()    

        memories = self.pipeline.model2.memory.to_dict()    
        memories_json = json.dumps(memories, default=str)
        try:
            c.execute("""
                INSERT INTO head_memory_storage 
                (memory_name, model_type, weights, is_active)
                VALUES (?, ?, ?, ?)
            """, (memory_name, model_type, memories_json, 1))
        
            c.execute("""
                UPDATE head_memory_storage 
                SET is_active = 0 
                WHERE memory_name = ? AND id != last_insert_rowid()
            """, (memory_name,)) 
            
            conn.commit()
            conn.close()

            print('[||] All kNN-Memory heads dictionary saved!')

        except Exception as e:
            print(f'[-] Cant save Memory heads due to: {e}')
            pass    

    def load_hnsw_setup(self, memory_name):
        """Load weights from database. Returns True if found."""
        try:
            db_path = self.get_database_path()            
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()

        c.execute("""
        SELECT weights FROM hnsw_weight_storage 
        WHERE memory_name = ? AND is_active = 1
        """, (memory_name,))               
    
        result = c.fetchone()
        conn.close()
    
        if result:
            print('[+] HNSW Setup memory is found!')     
        else:
            print(f'[=] No saved weights for {memory_name}')
            return False

        try:
            weights = result[0]
            if isinstance(weights, str):
                weights = json.loads(weights)

            self.pipeline.model2.memory.hnsw_arch.dim = weights.get('dim', 0)
            self.pipeline.model2.memory.hnsw_arch.M = weights.get('M', 0)
            self.pipeline.model2.memory.hnsw_arch.ef_construction = weights.get('ef_construction')
            self.pipeline.model2.memory.hnsw_arch.metric = weights.get('metric', [])
            self.pipeline.model2.memory.hnsw_arch.seed = weights.get('seed', 0)
            self.pipeline.model2.memory.hnsw_arch.vectors = weights.get('vectors', [])
            self.pipeline.model2.memory.hnsw_arch.neighbors = weights.get('neighbors', [])
            self.pipeline.model2.memory.hnsw_arch.max_level_of = weights.get('max_level_of', None)
            self.pipeline.model2.memory.hnsw_arch.entry_point = weights.get('entry_point', None)
            self.pipeline.model2.memory.hnsw_arch.max_level = weights.get('max_level', 0)
            self.pipeline.model2.memory.hnsw_arch._rng = weights.get('rng_state', 0)

            print('[+] All HNSW Setups are loaded!')

        except Exception as e:
            print(f'[!] Cant load any HNSW Setups due to: {e}')
            traceback.print_exc()     

    def load_head_memory_setup(self, memory_name):
        """Load weights from database. Returns True if found."""
        try:
            db_path = self.get_database_path()            
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()

        c.execute("""
        SELECT weights FROM head_memory_storage 
        WHERE memory_name = ? AND is_active = 1
        """, (memory_name,))               
    
        result = c.fetchone()
        conn.close()
    
        if result:
            print('[+] Memory head setup is found!')       
        else:
            print(f'[=] No saved weights for {memory_name}')
            return False

        try:
            weights = result[0]
            if isinstance(weights, str):
                weights = json.loads(weights)
            
            self.pipeline.model2.memory.n_heads = weights.get('n_heads', 0)
            self.pipeline.model2.memory.head_dim = weights.get('head_dim',0)
            self.pipeline.model2.memory.capacity = weights.get('capacity', 0)
            self.pipeline.model2.memory.k = weights.get('k', 0)
            self.pipeline.model2.memory.metric = weights.get('metric', [])
            self.pipeline.model2.memory.evict_batch = weights.get('evict_batch', [])
            self.pipeline.model2.memory.seed = weights.get('seed', None)
            self.pipeline.model2.memory._head_seeds = weights.get('head_seeds', [])
            self.pipeline.model2.memory.indices = weights.get('indices', [])
            self.pipeline.model2.memory.values = weights.get('values')
            self.pipeline.model2.memory._count = weights.get('count')

            print('[+] All Per-Head Memory Setups are loaded!')

        except Exception as e:
            print(f'[!] Cant load any Weights due to: {e}')
            traceback.print_exc()

    def _serialize_mlp(self, mlp):
        arrays, meta = {}, []
        for i, l in enumerate(mlp.layers):
            if l.W is None:                       # lazily initialised
                meta.append({'input_size': l.input_size, 'output_size': l.output_size, 'has_W': False})
                continue
            arrays[f'W{i}'], arrays[f'b{i}'] = l.W, l.b
            meta.append({'input_size': int(l.W.shape[0]), 'output_size': int(l.W.shape[1]), 'has_W': True})
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        to_list = lambda a: None if a is None else np.asarray(a, dtype=np.float64).tolist()
        return {
            'mlp_layer_meta':  meta,
            'mlp_layer_blob':  base64.b64encode(buf.getvalue()).decode('ascii'),
            'mlp_error_counts': to_list(mlp.error_counts),
            'mlp_pred_counts':  to_list(mlp.pred_counts),
            'mlp_error_decay':  None if mlp.error_decay is None else float(mlp.error_decay),
        }

    def _restore_mlp(self, mlp, payload):
        """Copy saved arrays INTO the existing layer objects. Never replaces them."""
        meta = payload.get('mlp_layer_meta') or []
        if not meta or not mlp.layers:
            return False, 'no saved layers, or model not built yet'
        if len(meta) != len(mlp.layers):
            return False, f'layer count {len(meta)} != {len(mlp.layers)}'

        data = np.load(io.BytesIO(base64.b64decode(payload['mlp_layer_blob'])), allow_pickle=False)
        staged = []
        for i, (mt, layer) in enumerate(zip(meta, mlp.layers)):
            if not mt['has_W']:
                staged.append(None); continue
            W, b = data[f'W{i}'], data[f'b{i}']
            if layer.output_size != W.shape[1] or layer.input_size not in (None, W.shape[0]):
                return False, f'layer {i}: saved {W.shape} vs built ({layer.input_size},{layer.output_size})'
            staged.append((W, b))

        for layer, wb in zip(mlp.layers, staged):        # all-or-nothing commit
            if wb is not None:
                layer.W, layer.b = wb[0].copy(), wb[1].copy()
                layer.input_size = layer.W.shape[0]
                layer.opt = None                         # forward() re-creates the optimizer
        for k in ('error_counts', 'pred_counts'):
            v = payload.get('mlp_' + k)
            setattr(mlp, k, None if v is None else np.asarray(v, dtype=np.float64))
        if payload.get('mlp_error_decay') is not None:
            mlp.error_decay = payload['mlp_error_decay']
        return True, 'ok'


    def load_mlp_weights(self, memory_name: str):
        """Load the latest active 'mlp' weights row for `memory_name`. Returns (ok, payload); (False, None) when nothing usable is stored."""
        try:
            conn = sqlite3.connect(self.get_database_path())
        except Exception:
            conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("""SELECT weights FROM weight_storage
                        WHERE memory_name = ? AND model_type = 'mlp' AND is_active = 1
                        ORDER BY id DESC LIMIT 1""", (memory_name,))
            result = c.fetchone()
        finally:
            conn.close()

        if not result:
            print(f'[=] No saved MLP weights for {memory_name}')
            return False, None
        try:
            payload = result[0]
            if isinstance(payload, str):
                payload = json.loads(payload)
            mlp = self.pipeline.model3
            if mlp is None:
                print('[>] model3 is not built yet -> skipping MLP weight restore.')
                return False, None
            if 'mlp_layer_meta' not in payload:
                print('[>] Legacy MLP row (layer represent strings, no weights) -> MLP Weights will not be loaded.')
                return False, None
            ok, msg = self._restore_mlp(mlp, payload)
            print('[+] All MLP Weights are loaded!' if ok else f'[>] MLP Weights not loaded: {msg}')
            return ok, payload
        except Exception as e:
            print(f'[!] Cant load MLP Weights due to: {e}')
            traceback.print_exc()
            return False, None

    def save_weights(self, memory_name, model_type=None):
        """Save weights to database."""
        try:
            db_path = self.get_database_path()
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()

        weights = {
            'lstm_W'  : self.pipeline.network_model.cell.W.tolist(),
            'lstm_b'  : self.pipeline.network_model.cell.b.tolist(),
            'Wy' : self.pipeline.network_model.Wy.tolist() if self.pipeline.network_model.Wy is not None else None,
            'by' : self.pipeline.network_model.by.tolist(),
            'residual_mean': self.pipeline.lstm_engine.residual_mean,
            'residual_std' : self.pipeline.lstm_engine.residual_std,
            'quantiles'    : {str(k): list(v) 
                            for k, v in self.pipeline.lstm_engine.quantiles.items()} 
                            if self.pipeline.lstm_engine.quantiles else {},
            'n_samples'    : self.pipeline.lstm_engine.n_samples,
            'saved_at'     : datetime.now().isoformat(),
        }
        
        weight_json = json.dumps(weights, default=str)
        if _RUST_MODULE_AVAILABLE:
            try:
                wc.save_lstm_weights(self.db_path, memory_name, weight_json)
                print('[||] LSTM weights saved using Rust module !')
            except Exception as e:
                print(f'[!] Rust save failed, falling back to Python: {e}')
        try:
            c.execute("""
                INSERT INTO weight_storage 
                (memory_name, model_type, weights, is_active)
                VALUES (?, ?, ?, ?)
            """, (memory_name, model_type, weight_json, 1))
        
            c.execute("""UPDATE weight_storage SET is_active = 0
            WHERE memory_name = ? AND id != last_insert_rowid()
            AND COALESCE(model_type,'') NOT IN ('mlp','transformer')""", (memory_name,))

            conn.commit()
            conn.close()

            self.save_transformer_weights(memory_name)
            self.save_mlp_weights(self.pipeline.model3, memory_name)
            print('[||] All Weights dictionary saved!')

        except Exception as e:
            print(f'[-] Cant save Weights due to: {e}')
            pass   
        

    def save_mlp_weights(self, mlp: Any, memory_name: str, model_type='mlp'):
        """Save the given MLP's layers/weights under `memory_name`."""
        mlp = self.pipeline.model3

        try:
            db_path = self.get_database_path()
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()

        weights = {
            'mlp_layers' : mlp.layers,
            'mlp_error_counts' : mlp.error_counts,
            'mlp_pred_counts'  : mlp.pred_counts,
            'mlp_error_decay'  : mlp.error_decay,
            
        }
        weight_json = json.dumps(self._serialize_mlp(mlp))
        try:
            c.execute("""
                INSERT INTO weight_storage 
                (memory_name, model_type, weights, is_active)
                VALUES (?, ?, ?, ?)
            """, (memory_name, model_type, weight_json, 1))
        
            c.execute("""UPDATE weight_storage SET is_active = 0
             WHERE memory_name = ? AND model_type = 'mlp'
               AND id != last_insert_rowid()""", (memory_name,))
            
            conn.commit()
            conn.close()

            print('[||] MLP Weights dictionary saved!')

        except Exception as e:
            print(f'[-] Cant save Weights due to: {e}')
            pass    


    def save_transformer_weights(self, memory_name: str):
        """Save transformer weights as compressed binary blob."""
        tf = self.pipeline.model2
        try:
            db_path = self.get_database_path()
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()

        buf = io.BytesIO()
        np.savez_compressed(buf,
            token_embedding = tf.token_embedding,
            pos_embedding   = tf.pos_embedding,
            W_q             = tf.W_q,
            W_k             = tf.W_k,
            W_v             = tf.W_v,
            proj            = tf.input_proj,
            proj_bias       = tf.input_proj_bias,
            W_q_fixed       = tf.W_q_fixed,
            W_k_fixed       = tf.W_k_fixed,
            W_v_fixed       = tf.W_v_fixed,
            W_o             = tf.W_o,
            ffn1            = tf.ffn1,
            ffn2            = tf.ffn2,
            ln1_scale       = tf.ln1_scale,
            ln1_shift       = tf.ln1_shift,
            ln2_scale       = tf.ln2_scale,
            ln2_shift       = tf.ln2_shift,
            output          = tf.output,
            output_bias     = tf.output_bias
        )
        binary_data = buf.getvalue()
        if _RUST_MODULE_AVAILABLE:
            try:
                wc.save_transformer_weights(self.db_path, memory_name, binary_data)
                print('[||] LSTM weights saved using Rust module for flexibility!')
                return
            except Exception as e:
                print(f'[!] Rust save failed, falling back to Python: {e}')
        else:
            print('[=] Rust module unavailable, using python sqlite3.')

        try:
            c.execute("""
                INSERT INTO weight_storage
                (memory_name, model_type, weights, is_active)
                VALUES (?, ?, ?, ?)
            """, (memory_name, 'transformer', 
                sqlite3.Binary(binary_data), 1))

            c.execute("""
                UPDATE weight_storage SET is_active = 0
                WHERE memory_name = ?
                AND model_type = 'transformer'
                AND id != last_insert_rowid()
            """, (memory_name,))


            conn.commit()
            print('[||] Transformer weights saved!')

        except Exception as e:
            print(f'[!] Transformer weight save failed: {e}')
            conn.rollback()
        finally:
            conn.close()

    def load_transformer_weights(self, memory_name: str) -> bool:
        """Restore the Transformer's weights (including its target memory) from storage; returns True on success."""
        try:
            db_path = self.get_database_path()
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor() 


        try:
            if _RUST_MODULE_AVAILABLE:
                try:
                    binary = wc.load_transformer_weights(self.db_path, memory_name) 

                    buf = io.BytesIO(bytes(binary))
                    data = np.load(buf, allow_pickle=True)

                    print('[+] Transformer weights data loaded using Rust module!')

                    return True, data

                except Exception as e:
                    print(f'[=] Cant load Transformer weights: {e}, using python sqlite3 to handle weights.')
                    c.execute("""
                        SELECT weights FROM weight_storage
                        WHERE memory_name = ? AND model_type = 'transformer' AND is_active = 1
                        ORDER BY id DESC LIMIT 1
                    """, (memory_name,))
                    row = c.fetchone()
                    if not row:
                        print(f'[=] No saved transformer weights for {memory_name}')
                        return False, None

                    buf = io.BytesIO(bytes(row[0]))
                    data = np.load(buf, allow_pickle=True)

            else:
                c.execute("""
                    SELECT weights FROM weight_storage
                    WHERE memory_name = ? AND model_type = 'transformer' AND is_active = 1
                    ORDER BY id DESC LIMIT 1
                """, (memory_name,))
                row = c.fetchone()
                if not row:
                    print(f'[=] No saved transformer weights for {memory_name}')
                    return False, None

                buf = io.BytesIO(bytes(row[0]))
                data = np.load(buf, allow_pickle=True)

            t = self.pipeline.model2
            if data['W_q'] is not None and len(data['W_q']) > 0:
                t.token_embedding = data['token_embedding']
                t.pos_embedding   = data['pos_embedding']
                t.W_q             = data['W_q']
                t.W_k             = data['W_k']
                t.W_v             = data['W_v']
                t.input_proj      = data['proj']
                t.input_proj_bias = data['proj_bias']
                t.W_q_fixed       = data['W_q_fixed']
                t.W_k_fixed       = data['W_k_fixed']
                t.W_v_fixed       = data['W_v_fixed']
                t.W_o             = data['W_o']
                t.ffn1            = data['ffn1']
                t.ffn2            = data['ffn2']
                t.ln1_scale       = data['ln1_scale']
                t.ln1_shift       = data['ln1_shift']
                t.ln2_scale       = data['ln2_scale']
                t.ln2_shift       = data['ln2_shift']
                t.output          = data['output']
                t.output_bias     = data['output_bias']

                print(f'[||] Transformer weights loaded!')
            else:
                print('[>] Transformer weights are not loaded due to incomplete data.')
            return True, data

        except Exception as e:
            print(f'[!] Transformer weight load failed: {e}')
            return False, None
        finally:
            conn.close()


    def load_weights(self, memory_name):
        """Load weights from database. Returns True if found."""
        if _RUST_MODULE_AVAILABLE:
            try:
                result = wc.load_lstm_weights(self.db_path, memory_name)
                print('[+] LSTM weights loaded via Rust module')
            except Exception as e:
                result = self.weight_retrieval(memory_name)
                print(f'[=] Cant load LSTM Weights due to: {e}, using python sqlite3 as fallback.')
                  
        else:
            result = self.weight_retrieval(memory_name)           
    
        if not result:
            print(f'[=] No saved weights for {memory_name}')
            return False, None, None, None

        try:
            weights = result
            if isinstance(weights, str):
                weights = json.loads(weights)

            if weights.get('lstm_W') is not None:
                if len(np.array(weights.get('lstm_W'))) > 0:
                    self.pipeline.network_model.cell.W  = np.array(weights.get('lstm_W'))
                    self.pipeline.network_model.cell.b  = np.array(weights.get('lstm_b'))
                    self.pipeline.network_model.Wy      = np.array(weights.get('Wy')) if weights.get('Wy') else None
                    self.pipeline.network_model.by      = np.array(weights.get('by'))
                    if self.pipeline.lstm_engine is not None:
                        if self.pipeline.lstm_engine.residual_mean is not None:
                            self.pipeline.lstm_engine.residual_mean = weights.get('residual_mean', 0.0)
                        if self.pipeline.lstm_engine.residual_std is not None:
                            self.pipeline.lstm_engine.residual_std  = weights.get('residual_std',  1.0)
                        if self.pipeline.lstm_engine.n_samples:
                            self.pipeline.lstm_engine.n_samples     = weights.get('n_samples', self.pipeline.lstm_engine.n_samples)
                        if self.pipeline.lstm_engine.quantiles:
                            self.pipeline.lstm_engine.quantiles     = {float(k): tuple(v) 
                                            for k, v in weights.get('quantiles', {}).items()}
                    print('[MEMORY] LSTM Weights are loaded properly!')
                else:
                    print('[MEMORY] LSTM Memory is None, cannot load any Memory.')
            else:
                print('[>] LSTM Weights are not loaded due to incomplete data.')

            tf_loaded, tf_weights = self.load_transformer_weights(memory_name)
            mlp_loaded, mlp_weights = self.load_mlp_weights(memory_name)
            print(f'[=] Transformer weights loaded: {tf_loaded}')
            if tf_loaded and mlp_loaded:
                print(f'[=] Weights loaded for {memory_name}  '
                    f'(saved at {weights.get("saved_at", "unknown")})')
            return True, weights, tf_weights, mlp_loaded

        except Exception as e:
            print(f'[!] Cant load any Weights due to: {e}')
            traceback.print_exc()

            return False, None, None, None


    def load_transformer_dict(self, memory_name):
        """Load the stored Transformer dict (weights and config) for `memory_name`."""
        try:
            try:
                conn = sqlite3.connect(self.db_path)
            except:
                db_path = self.get_database_path()
                conn = sqlite3.connect(db_path)   

            if _RUST_MODULE_AVAILABLE:
                try:
                    data = wc.load_attention_dict(self.db_path, memory_name)
                    print('[+] Transformer attention loaded using Rust module!')
                    return data
                except Exception as e:
                    print(f'[=]: {e}, Loading transformer attention from python sqlite3...')
            else:     
                conn = sqlite3.connect(self.db_path)
                c = conn.cursor()
            
                c.execute("""
                SELECT model_data FROM model_attn_storage 
                WHERE memory_name = ? AND is_active = 1
                """, (memory_name,))
            
                result = c.fetchone()
                conn.close()
            
                if result:
                    return json.loads(result[0])
        except Exception as e:
            print(f'[!] Error handling attention dict: {e}')

        return None   

    def save_peer_needs_dict(self, memory_name, model_dict, target_pred, agent_id, model_type='Pipeline'):
        """Store the model dict and target predictions a peer agent needs (keyed by memory and agent id)."""
        try:
            db_path = self.get_database_path()
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()

        model_json = json.dumps(model_dict, default=str)
        target_json = json.dumps(target_pred, default=str)
        agent_id_converted = json.dumps(agent_id, default=str)

        try:
            c.execute("""
                INSERT INTO agent_attn_storage 
                (memory_name, model_type, model_attn_data, model_target_pred, agent_id, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (memory_name, model_type, model_json, target_json, agent_id_converted, 1))
        
            c.execute("""
                UPDATE agent_attn_storage 
                SET is_active = 0 
                WHERE memory_name = ? AND id != last_insert_rowid()
            """, (memory_name,)) 

            conn.commit()
            conn.close()

            print('|| Peer data Needs dictionary saved!')

        except Exception as e:
            print(f'[-] Cant save model memory due to: {e}') 
            pass        


    def load_peer_request_dict(self, memory_name, agent_id):
        """Load the data stored for a peer agent's request."""
        print(f'|| Peer request with Agent')
        try:
            try:
                db_path = self.get_database_path()
                conn = sqlite3.connect(db_path)   
            except:
                conn = sqlite3.connect(self.db_path)


            c = conn.cursor()
            placeholders = ",".join(["?"] * len(agent_id))

            query = f"""
            SELECT model_attn_data, model_target_pred FROM agent_attn_storage 
            WHERE memory_name = ? AND is_active = 1 AND agent_id NOT IN ({placeholders})
            """
            params = [memory_name] + agent_id
        
            c.execute(query, params)  
        
            result = c.fetchone()
            conn.close()
            print(f"|| Retrieved Peer Request memory: {memory_name} for agent_id: {agent_id}: with total result: {len(result)}")
 
            if result:
                return json.loads(result[0]), json.loads(result[1])
            return None, None
        except Exception as e:
            print(f'|| Cant load peer request memory due to: {e}') 
            return None, None  

        

    def fix_corrupted_memory(self, memory_name):
        """Deactivate corrupted entries for `memory_name` so they are not loaded again."""
        # Clear corrupted memory entries
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Deactivate corrupted entries
            c.execute("""
                UPDATE model_storage 
                SET is_active = 0 
                WHERE memory_name = ? AND is_active = 1
            """, (memory_name,))
            
            c.execute("""
                UPDATE model_attn_storage 
                SET is_active = 0 
                WHERE memory_name = ? AND is_active = 1
            """, (memory_name,))
            
            conn.commit()
            conn.close()
            
            print(f'[✅] Cleared corrupted memory for {memory_name}')
            return True
        except Exception as e:
            print(f'[!] Failed to clear memory: {e}')
            return False


    def load_agent_id(self, memory_name):
        """Load the agent id associated with `memory_name`."""
        try:

            try:
               db_path = self.get_database_path()
               conn = sqlite3.connect(db_path)  
            except:
                conn = sqlite3.connect(self.db_path)

            if _RUST_MODULE_AVAILABLE:
                try:
                    agent_data = wc.load_agent_id(self.db_path, memory_name)
                    print('[+] Got Agent ID Data from Database!')
                except Exception as e:
                    print(f'[!] Cant load agent ID data from DB: {e}')
                    return None    
            else:
                c = conn.cursor()
            
                c.execute("""
                SELECT agent_id FROM agent_attn_storage 
                WHERE memory_name = ? AND is_active = 1
                """, (memory_name,))
            
                result = c.fetchone()
                conn.close()

                print(f'[+] Retrieved Agent ID of {memory_name}: result: {result}')
            
                if result:
                    return json.loads(result[0])
        except Exception as e:
            print(f'[-] Error loading ID from database: {e}')

        return None        


    def memory_exists(self, memory_name, type=None):

        """Return True if a saved memory of the given type exists for `memory_name`, else False."""
        conn = None
        try:
            try:
                db_path = self.get_database_path()
                conn = sqlite3.connect(db_path)               
            except:
                conn = sqlite3.connect(self.db_path)

            if _RUST_MODULE_AVAILABLE:
                try:
                    exists = wc.verify_memory_exist(self.db_path, memory_name, type)
                    print(f'[=] Memory exist: {exists}')
                    return exists
                except:
                    pass
            else:
                if type == 'Transformer':
                    c = conn.cursor()
            
                    c.execute("""
                    SELECT 1 FROM model_attn_storage 
                    WHERE memory_name = ? AND is_active = 1
                    LIMIT 1
                    """, (memory_name,))
            
                    result = c.fetchone()
                    exists = result is not None
                    print(f"|| Retrieved Attention: {memory_name}")

                elif type == 'Peer':
                    c = conn.cursor()
            
                    c.execute("""
                    SELECT 1 FROM agent_attn_storage 
                    WHERE memory_name = ? AND is_active = 1
                    LIMIT 1
                    """, (memory_name,))
            
                    result = c.fetchone()
                    exists = result is not None
                    print(f"|| Retrieved Peer Memory: {memory_name}")

                elif type == 'Accurate-Cache':
                    c = conn.cursor()

                    c.execute("""
                    SELECT 1 FROM accurate_cache_storage
                    WHERE memory_name = ? and is_active = 1
                    LIMIT 1""", (memory_name, ))

                    result = c.fetchone()
                    exists = result is not None
                    print(f"|| Retrieved Accurate Fact Cache Memory for memory: {memory_name}")
                else:
                    c = conn.cursor()

                    c.execute("""
                    SELECT 1 FROM model_storage 
                    WHERE memory_name = ? AND is_active = 1
                    LIMIT 1
                    """, (memory_name,))
            
                    result = c.fetchone()
                    exists = result is not None
                    print(f"|| Retrieved Memory: {memory_name}")

                return exists
        
        except sqlite3.OperationalError as e:
            print(f"[!] Database error: {e}")
            return False
            
        except Exception as e:
            print(f"[!] Unexpected error in handling memory: {e}") 
            return False
        finally:
            if conn:
                conn.close()


    def save_model_binary(self, model_object, memory_name, model_type='mlp'):
        """Store a model object as a binary blob (serialised with joblib) in model_storage under `memory_name` / `model_type`."""
        try:
            try:
                conn = sqlite3.connect(self.db_path)
            except:
               db_path = self.get_database_path()
               conn = sqlite3.connect(db_path)          
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
        
            model_binary = joblib.dumps(model_object)
        
            c.execute("""
            INSERT INTO model_storage 
            (memory_name, model_type, model_binary, is_active)
            VALUES (?, ?, ?, ?)
            """, (memory_name, model_type, model_binary, 1))
        
            # Deactivate other versions
            c.execute("""
            UPDATE model_storage 
            SET is_active = 0 
            WHERE memory_name = ? AND id != last_insert_rowid()
            """, (memory_name,))
        
            conn.commit()
            model_id = c.lastrowid
            print(f"✅ Memory '{memory_name}' saved as binary (ID: {model_id})")
        except Exception as e:
            logger.error(f"[-] Error handling: {e}")

        conn.close()

        return model_id
    
    def load_model_binary(self, memory_name):
        """Load and deserialise a model object saved with save_model_binary."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
            SELECT model_binary FROM model_storage 
            WHERE memory_name = ? AND is_active = 1
        """, (memory_name,))
        
        result = c.fetchone()
        conn.close()
        
        if result:
            return joblib.loads(result[0])
        return None
    
    def save_complete_pipeline(self, pipeline_name, pipeline_dict):
        """Store the whole pipeline state (as JSON) under `pipeline_name`."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Convert entire pipeline to JSON (for dicts)
        pipeline_json = json.dumps(pipeline_dict, default=str)
        
        c.execute("""
            INSERT INTO model_storage 
            (pipeline_name, model_type, model_data, metadata, is_active)
            VALUES (?, ?, ?, ?, ?)
        """, (pipeline_name, 'pipeline', pipeline_json, 
               json.dumps({'components': list(pipeline_dict.keys())}), 1))
        
        conn.commit()
        model_id = c.lastrowid
        conn.close()
        
        print(f"✅ Integrated pipeline '{pipeline_name}' saved")
        return model_id
