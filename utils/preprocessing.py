import pandas as pd
import numpy as np
import difflib
from sklearn.preprocessing import StandardScaler, OneHotEncoder

TARGET_CANDIDATES = ['Global_Sales', 'Global_sales', 'global_sales', 'GlobalSales']
LEAK_COLUMNS = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Rank', 'Name']

TOP_PUBLISHER_COUNT = 25
TOP_PLATFORM_COUNT = 20

COLUMN_ALIASES = {
    'Platform': ['platform', 'console', 'system', 'platform_name', 'device'],
    'Genre': ['genre', 'category', 'game_genre', 'type', 'game_type'],
    'Publisher': ['publisher', 'publisher_name', 'company', 'label', 'studio'],
    'Year': ['year', 'release_year', 'releaseyear', 'release_date', 'releasedate',
             'year_of_release', 'launch_year', 'yr']
}

REQUIRED_COLUMNS = ['Platform', 'Genre', 'Publisher', 'Year']


def _normalize_name(col):
    return str(col).strip().lower().replace(' ', '_').replace('-', '_')


def resolve_uploaded_columns(df):
    df = df.copy()
    normalized_lookup = {_normalize_name(c): c for c in df.columns}
    mapping_info = {}

    for target, aliases in COLUMN_ALIASES.items():
        if target in df.columns:
            continue

        found_original = None
        for alias in aliases:
            if alias in normalized_lookup:
                found_original = normalized_lookup[alias]
                break

        if not found_original:
            close = difflib.get_close_matches(target.lower(), normalized_lookup.keys(), n=1, cutoff=0.72)
            if close:
                found_original = normalized_lookup[close[0]]

        if found_original:
            df = df.rename(columns={found_original: target})
            mapping_info[target] = found_original

    return df, mapping_info


def extract_year_column(series):
    numeric = pd.to_numeric(series, errors='coerce')
    if numeric.notna().sum() >= series.notna().sum() * 0.5:
        return numeric
    parsed_dates = pd.to_datetime(series, errors='coerce')
    return parsed_dates.dt.year


def fill_missing_required_columns(df, defaults):
    warnings = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = defaults.get(col, 'Unknown')
            warnings.append(col)
    return df, warnings


def normalize_uploaded_dataframe(df, defaults=None):
    defaults = defaults or {}
    df, mapping_info = resolve_uploaded_columns(df)

    if 'Year' in df.columns:
        df['Year'] = extract_year_column(df['Year'])

    df, defaulted_columns = fill_missing_required_columns(df, defaults)

    return df, mapping_info, defaulted_columns


def detect_target_column(df):
    for name in TARGET_CANDIDATES:
        if name in df.columns:
            return name
    for col in df.columns:
        if 'global' in col.lower() and 'sale' in col.lower():
            return col
    raise ValueError('could not find a global sales column in this dataset')


def load_raw_dataset(path):
    df = pd.read_csv(path)
    return df


def clean_dataset(df):
    df = df.copy()
    df = df.drop_duplicates()

    target = detect_target_column(df)

    if 'Year' in df.columns:
        median_year = df['Year'].median()
        df['Year'] = df['Year'].fillna(median_year)
        df['Year'] = df['Year'].astype(int)

    for col in ['Publisher', 'Developer', 'Genre', 'Platform', 'Rating']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    for col in ['Critic_Score', 'User_Score']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    q1 = df[target].quantile(0.25)
    q3 = df[target].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr
    df[target] = np.where(df[target] > upper_bound, upper_bound, df[target])

    drop_cols = [c for c in LEAK_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df, target


def make_sales_category(df, target, n_bins=3):
    labels = ['Low', 'Medium', 'High'] if n_bins == 3 else [str(i) for i in range(n_bins)]
    category = pd.qcut(df[target], q=n_bins, labels=labels, duplicates='drop')
    return category.astype(str)


def reduce_high_cardinality(series, top_n):
    counts = series.value_counts()
    keep = set(counts.head(top_n).index)
    return series.apply(lambda x: x if x in keep else 'Other')


def build_feature_frame(df, target, fit_mode=True, encoders=None, scaler=None, feature_columns=None):
    df = df.copy()

    categorical_cols = [c for c in ['Platform', 'Genre', 'Publisher', 'Developer', 'Rating'] if c in df.columns]
    numeric_cols = [c for c in ['Year', 'Critic_Score', 'User_Score'] if c in df.columns]

    if 'Publisher' in df.columns:
        if fit_mode:
            top_publishers = df['Publisher'].value_counts().head(TOP_PUBLISHER_COUNT).index.tolist()
            encoders = encoders or {}
            encoders['top_publishers'] = top_publishers
        else:
            top_publishers = encoders['top_publishers']
        df['Publisher'] = df['Publisher'].apply(lambda x: x if x in top_publishers else 'Other')

    if 'Platform' in df.columns:
        if fit_mode:
            top_platforms = df['Platform'].value_counts().head(TOP_PLATFORM_COUNT).index.tolist()
            encoders = encoders or {}
            encoders['top_platforms'] = top_platforms
        else:
            top_platforms = encoders['top_platforms']
        df['Platform'] = df['Platform'].apply(lambda x: x if x in top_platforms else 'Other')

    encoded_parts = []
    if fit_mode:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded = ohe.fit_transform(df[categorical_cols])
        encoders = encoders or {}
        encoders['ohe'] = ohe
        encoders['categorical_cols'] = categorical_cols
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(categorical_cols), index=df.index)
    else:
        ohe = encoders['ohe']
        encoded = ohe.transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(categorical_cols), index=df.index)

    numeric_df = df[numeric_cols] if numeric_cols else pd.DataFrame(index=df.index)

    if fit_mode:
        scaler = StandardScaler()
        if numeric_cols:
            scaled_values = scaler.fit_transform(numeric_df)
            numeric_df = pd.DataFrame(scaled_values, columns=numeric_cols, index=df.index)
    else:
        if numeric_cols and scaler is not None:
            scaled_values = scaler.transform(numeric_df)
            numeric_df = pd.DataFrame(scaled_values, columns=numeric_cols, index=df.index)

    final_df = pd.concat([numeric_df, encoded_df], axis=1)

    if not fit_mode and feature_columns is not None:
        for col in feature_columns:
            if col not in final_df.columns:
                final_df[col] = 0
        final_df = final_df[feature_columns]

    if fit_mode:
        return final_df, encoders, scaler, final_df.columns.tolist()
    return final_df


def prepare_training_data(dataset_path):
    df = load_raw_dataset(dataset_path)
    df, target = clean_dataset(df)
    category = make_sales_category(df, target)
    features, encoders, scaler, feature_columns = build_feature_frame(df, target, fit_mode=True)
    return {
        'raw_clean_df': df,
        'target': target,
        'features': features,
        'labels_regression': df[target].reset_index(drop=True),
        'labels_classification': category.reset_index(drop=True),
        'encoders': encoders,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'median_year': int(df['Year'].median()) if 'Year' in df.columns else 2010
    }


def prepare_inference_frame(input_df, encoders, scaler, feature_columns):
    df = input_df.copy()
    for col in ['Publisher', 'Developer', 'Genre', 'Platform', 'Rating']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    for col in ['Critic_Score', 'User_Score']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(df['Year'].median() if df['Year'].notna().any() else 2010)

    if 'Publisher' in df.columns:
        top_publishers = encoders['top_publishers']
        df['Publisher'] = df['Publisher'].apply(lambda x: x if x in top_publishers else 'Other')
    if 'Platform' in df.columns:
        top_platforms = encoders['top_platforms']
        df['Platform'] = df['Platform'].apply(lambda x: x if x in top_platforms else 'Other')

    categorical_cols = encoders['categorical_cols']
    ohe = encoders['ohe']
    encoded = ohe.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(categorical_cols), index=df.index)

    numeric_cols = [c for c in ['Year', 'Critic_Score', 'User_Score'] if c in df.columns]
    numeric_df = df[numeric_cols] if numeric_cols else pd.DataFrame(index=df.index)
    if numeric_cols:
        scaled_values = scaler.transform(numeric_df)
        numeric_df = pd.DataFrame(scaled_values, columns=numeric_cols, index=df.index)

    final_df = pd.concat([numeric_df, encoded_df], axis=1)
    for col in feature_columns:
        if col not in final_df.columns:
            final_df[col] = 0
    final_df = final_df[feature_columns]
    return final_df
