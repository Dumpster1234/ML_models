import pandas as pd
import numpy as np
from django.shortcuts import render, redirect
from .forms import DatasetUploadForm
from .models import Dataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeCV
import tempfile


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def upload_dataset(request):
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = Dataset(file=request.FILES['file'])
            dataset.save()
            return redirect('model_development', dataset_id=dataset.id)
    else:
        form = DatasetUploadForm()
    return render(request, 'ml_app/upload.html', {'form': form})

def model_development(request, dataset_id):
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        df = pd.read_csv(dataset.file.path)
        columns = df.columns.tolist()
        
        # Auto-detect target column
        target_names = ['target', 'label', 'class', 'y']
        target_column = None
        for col in columns:
            if col.lower() in target_names:
                target_column = col
                break
        if not target_column:
            target_column = columns[-1]  # Default to last column
        
        # Determine target type
        unique_values = df[target_column].dropna().unique()
        target_type = 'classification' if df[target_column].dtype in [np.int32, np.int64] or len(unique_values) <= 10 else 'regression'
        print("Detected Target:", target_column, "Type:", target_type)  # Debug
        
        # Store target in session
        request.session['target_column'] = target_column
        request.session['target_type'] = target_type
    except (Dataset.DoesNotExist, FileNotFoundError):
        return redirect('upload_dataset')
    
    selected_model = request.GET.get('model_type', '')
    print("Selected Model (GET):", selected_model)  # Debug
    
    if request.method == 'POST':
        selected_model = request.POST.get('model_type', selected_model)
        print("Selected Model (POST):", selected_model, "POST Data:", request.POST)  # Debug
        if not selected_model:
            return render(request, 'ml_app/model_development.html', {
                'columns': columns,
                'dataset_id': dataset_id,
                'selected_model': selected_model,
                'target_column': target_column,
                'target_type': target_type,
                'error': 'Please select a model from the sidebar.'
            })
        if (selected_model.startswith('classification') and target_type != 'classification') or \
           (selected_model.startswith('regression') and target_type != 'regression'):
            return render(request, 'ml_app/model_development.html', {
                'columns': columns,
                'dataset_id': dataset_id,
                'selected_model': selected_model,
                'target_column': target_column,
                'target_type': target_type,
                'error': f'Invalid model type: {selected_model} is not compatible with target column "{target_column}" ({target_type}).'
            })
        request.session['model_type'] = selected_model
        request.session['dataset_id'] = dataset_id
        return redirect('results')
    
    return render(request, 'ml_app/model_development.html', {
        'columns': columns,
        'dataset_id': dataset_id,
        'selected_model': selected_model,
        'target_column': target_column,
        'target_type': target_type,
    })

def results(request):
    print("Session Data:", {
        'dataset_id': request.session.get('dataset_id'),
        'model_type': request.session.get('model_type'),
        'target_column': request.session.get('target_column'),
        'target_type': request.session.get('target_type'),
    })
    
    dataset_id = request.session.get('dataset_id')
    model_type = request.session.get('model_type')
    target_column = request.session.get('target_column')
    target_type = request.session.get('target_type')
    
    if not all([dataset_id, model_type, target_column, target_type]):
        return render(request, 'ml_app/results.html', {
            'error': 'Session data is incomplete. Please upload a dataset and configure the model.',
            'dataset_id': dataset_id or 0,
        })
    
    
    # Load dataset
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        df = pd.read_csv(dataset.file.path)
        print("Dataset Columns:", df.columns.tolist())
    except (Dataset.DoesNotExist, FileNotFoundError) as e:
        return render(request, 'ml_app/results.html', {
            'error': f'Failed to load dataset: {str(e)}',
            'dataset_id': dataset_id,
        })
    
    # Validate data
    if target_column not in df.columns:
        return render(request, 'ml_app/results.html', {
            'error': f'Target column "{target_column}" not found in dataset.',
            'dataset_id': dataset_id,
        })
    
    # Prepare data
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Validate numeric features
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        if not numeric_columns.tolist():
            return render(request, 'ml_app/results.html', {
                'error': 'No numeric features found in dataset.',
                'dataset_id': dataset_id,
            })
        X = X[numeric_columns]
        
        # Handle NaNs
        if X.isnull().any().any() or y.isnull().any():
            print("NaNs detected in dataset")
            imputer = SimpleImputer(strategy='mean')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
            valid_indices = ~y.isnull()
            X = X.loc[valid_indices]
            y = y.loc[valid_indices]
            if X.empty or y.empty:
                return render(request, 'ml_app/results.html', {
                    'error': 'Dataset is empty after removing rows with missing target values.',
                    'dataset_id': dataset_id,
                })
        
        # Scale features for SVM and Neural Network
        if model_type in ['classification_svm', 'classification_nn']:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        # Handle non-numeric target for classification
        if model_type.startswith('classification'):
            if y.dtype not in [np.int32, np.int64] and len(y.unique()) > 10:
                return render(request, 'ml_app/results.html', {
                    'error': f'Target column "{target_column}" has continuous or too many unique values for classification.',
                    'dataset_id': dataset_id,
                })
            if y.dtype not in [np.int32, np.int64]:
                le = LabelEncoder()
                y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Features:", X.columns.tolist())
    except Exception as e:
        return render(request, 'ml_app/results.html', {
            'error': f'Data preparation failed: {str(e)}',
            'dataset_id': dataset_id,
        })
    
    # Define models and hyperparameter grids
    try:
        if model_type == 'classification_logistic':
            model = LogisticRegression(max_iter=1000)
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l2']
            }

        elif model_type == 'classification_nb':
            model = GaussianNB()
            param_grid = {
                'var_smoothing': [1e-9, 1e-8, 1e-7]
            }

        elif model_type == 'classification_ada':
            model = AdaBoostClassifier()
            param_grid = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.5, 1.0, 1.5]
            }
        elif model_type == 'classification_catboost':
            model = CatBoostClassifier(verbose=0)
            param_grid = {
                    'iterations': [100, 200],
                    'depth': [4, 6, 10],
                    'learning_rate': [0.03, 0.1]
                }

        # elif model_type == 'time_series_arima':
        #     # Placeholder for ARIMA or other time series model
        #     from statsmodels.tsa.arima.model import ARIMA
        #     model = ARIMA(endog=train_series, order=(1, 1, 1))  # Replace train_series
        #     param_grid = {}  # Manual grid search likely needed for ARIMA
        #     best_model = model.fit()
        #     # Skip GridSearchCV for ARIMA
        #     return best_model

       

        elif model_type == 'classification_rf':
            model = RandomForestClassifier()
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None]
            }
        elif model_type == 'classification_svm':
            model = SVC(max_iter=1000)
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf']
            }
        # elif model_type == 'classification_nn':
        #     model = MLPClassifier(max_iter=1000)
        #     param_grid = {
        #         'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        #         'alpha': [0.0001, 0.001]
        #     }
        elif model_type == 'classification_xgb':
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1]
            }

        elif model_type == 'classification_lgbm':

            model = LGBMClassifier()
            param_grid = {
                'num_leaves': [31, 50],
                'max_depth': [-1, 10, 20],
                'learning_rate': [0.1, 0.01],
                'n_estimators': [100, 200]
            }
            

        elif model_type == 'regression_linear':
            model = LinearRegression()
            param_grid = {}
        elif model_type == 'regression_lasso':
            model = Lasso()
            param_grid = {
                'alpha': [0.01, 0.1, 1.0, 10.0]
            }
        elif model_type == 'regression_ridge':
            model = Ridge()
            param_grid = {
                'alpha': [0.01, 0.1, 1.0, 10.0]
            }
        elif model_type == 'regression_dt':
            model = DecisionTreeRegressor()
            param_grid = {
                'max_depth': [3, 5, 10, None]
            }
        elif model_type == 'regression_gb':
            model = GradientBoostingRegressor()
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1]
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Perform GridSearchCV
        if param_grid:
            scoring = 'accuracy' if model_type.startswith('classification') else 'r2'
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1, error_score='raise')
            print("Starting GridSearchCV for", model_type, "with param_grid:", param_grid)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print("Best parameters:", best_params)
        else:
            print("Training model:", model_type, "with default parameters")
            model.fit(X_train, y_train)
            best_params = {}
        
        y_pred = model.predict(X_test)
        print("Model trained successfully")
    except Exception as e:
        return render(request, 'ml_app/results.html', {
            'error': f'Model training failed for {model_type}: {str(e)}',
            'dataset_id': dataset_id,
        })
    
    # Calculate metrics
    try:
        if model_type.startswith('classification'):
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'F1 Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            }
        else:
            metrics = {
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R² Score': r2_score(y_test, y_pred),
            }
        print("Metrics:", metrics)
    except Exception as e:
        return render(request, 'ml_app/results.html', {
            'error': f'Metrics calculation failed for {model_type}: {str(e)}',
            'dataset_id': dataset_id,
        })
    
    # Feature importance
    feature_importance = []
    try:
        if hasattr(model, 'feature_importances_'):
            feature_importance = [(str(name), float(imp)) for name, imp in zip(X.columns, model.feature_importances_)]
        elif hasattr(model, 'coef_'):
            feature_importance = [(str(name), float(abs(imp))) for name, imp in zip(X.columns, model.coef_)]
        print("Feature Importance:", feature_importance)
    except Exception as e:
        print("Feature Importance Error:", str(e))
    
    return render(request, 'ml_app/results.html', {
        'model_type': model_type,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'dataset_id': dataset_id,
        'best_params': best_params,
        'target_column': target_column,
    })