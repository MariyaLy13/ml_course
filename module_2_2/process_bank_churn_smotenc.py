import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, Any
from imblearn.over_sampling import SMOTENC

# ---------------------------------------------------------
# 1. Create inputs/targets
# ---------------------------------------------------------
def create_inputs_targets(train_df, val_df, input_cols, target_col):
    """Створює inputs/targets для train та val."""
    data = {
        'train_inputs': train_df[input_cols].copy(),
        'train_targets': train_df[target_col].copy(),
        'val_inputs': val_df[input_cols].copy(),
        'val_targets': val_df[target_col].copy()
    }
    return data


# ---------------------------------------------------------
# 2. Impute numeric
# ---------------------------------------------------------
def impute_missing_values(data: Dict[str, Any], numeric_cols: list):
    """Імп'ютинг числових колонок."""
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(data['train_inputs'][numeric_cols])

    for split in ['train', 'val']:
        data[f'{split}_inputs'][numeric_cols] = imputer.transform(
            data[f'{split}_inputs'][numeric_cols]
        )

    return imputer


# ---------------------------------------------------------
# 3. Scale numeric
# ---------------------------------------------------------
def scale_numeric_features(data: Dict[str, Any], numeric_cols: list):
    """Масштабування числових колонок."""
    scaler = MinMaxScaler()
    scaler.fit(data['train_inputs'][numeric_cols])

    for split in ['train', 'val']:
        data[f'{split}_inputs'][numeric_cols] = scaler.transform(
            data[f'{split}_inputs'][numeric_cols]
        )

    return scaler


# ---------------------------------------------------------
# 4. One-hot encode categorical
# ---------------------------------------------------------
def encode_categorical_features(data: Dict[str, Any], categorical_cols: list):
    """One-hot кодування категоріальних колонок."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(data['train_inputs'][categorical_cols])

    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    for split in ['train', 'val']:
        encoded = encoder.transform(data[f'{split}_inputs'][categorical_cols])
        encoded_df = pd.DataFrame(
            encoded,
            columns=encoded_cols,
            index=data[f'{split}_inputs'].index
        )

        data[f'{split}_inputs'] = pd.concat(
            [data[f'{split}_inputs'], encoded_df],
            axis=1
        )

        data[f'{split}_inputs'].drop(columns=categorical_cols, inplace=True)

    return encoder, encoded_cols


# ---------------------------------------------------------
# 5. Main preprocessing for training
# ---------------------------------------------------------
def preprocess_data(raw_df: pd.DataFrame):

    # --- 1. Train/val split ---
    train_df, val_df = train_test_split(
        raw_df,
        test_size=0.2,
        random_state=42,
        stratify=raw_df['Exited']
    )

    # --- 2. Вибір колонок ---
    input_cols = list(train_df.columns)[2:-1]
    target_col = 'Exited'

    # --- 3. Створення inputs/targets ---
    data = create_inputs_targets(train_df, val_df, input_cols, target_col)

    # --- 4. Визначення типів колонок ---
    numeric_cols = data['train_inputs'].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data['train_inputs'].select_dtypes(include='object').columns.tolist()

    # --- 5. Імп'ютинг ---
    imputer = impute_missing_values(data, numeric_cols)

    # --- 6. Масштабування ---
    scaler = scale_numeric_features(data, numeric_cols)

    # --- 7. SMOTENC oversampling ---
    cat_indices = [data['train_inputs'].columns.get_loc(col) for col in categorical_cols]

    smotenc = SMOTENC(
        categorical_features=cat_indices,
        random_state=42
    )

    X_resampled, y_resampled = smotenc.fit_resample(
        data['train_inputs'],
        data['train_targets']
    )

    # Оновлюємо train після oversampling
    data['train_inputs'] = X_resampled
    data['train_targets'] = y_resampled

    # --- 8. One-hot encoding ---
    encoder, encoded_cols = encode_categorical_features(data, categorical_cols)

    # --- 9. Формування фінальних X ---
    X_train = data['train_inputs']
    X_val = data['val_inputs']
    train_targets = data['train_targets']
    val_targets = data['val_targets']

    return (
        X_train, train_targets,
        X_val, val_targets,
        input_cols, numeric_cols, categorical_cols,
        imputer, scaler, encoder
    )

# ---------------------------------------------------------
# 6. Preprocess NEW DATA (test.csv, production data)
# ---------------------------------------------------------
def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: list,
    numeric_cols: list,
    categorical_cols: list,
    imputer: SimpleImputer,
    scaler: MinMaxScaler,
    encoder: OneHotEncoder
):
    """
    Обробка нових даних (наприклад test.csv) з використанням
    вже навчених імп'ютера, скейлера та енкодера.

    Повертає X_new — готовий DataFrame для передбачення.
    """

    df = new_df.copy()

    # 1. Вибираємо тільки ті колонки, які були у тренуванні
    df = df[input_cols].copy()

    # 2. Імп'ютинг числових колонок
    df[numeric_cols] = imputer.transform(df[numeric_cols])

    # 3. Масштабування числових колонок
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # 4. One-hot encoding категоріальних колонок
    encoded = encoder.transform(df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoded_cols,
        index=df.index
    )

    # 5. Додаємо закодовані колонки
    df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

    return df