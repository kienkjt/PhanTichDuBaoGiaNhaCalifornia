from logging import root
import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import seaborn as sns
import scipy.stats as stats
from datetime import datetime
from sklearn.ensemble import IsolationForest
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Khởi tạo giao diện
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Phân tích giá nhà California")
app.geometry("1100x750")
app.resizable(True, True)

# Biến toàn cục
data = None
model = None
preprocessor = None
X_selected = None
selected_features = []
feature_checkboxes = []
checkbox_vars = []
categorical_features = []
numeric_features = []

# Hàm chuyển đổi chủ đề
def toggle_theme():
    current_mode = ctk.get_appearance_mode()
    new_mode = "Dark" if current_mode == "Light" else "Light"
    ctk.set_appearance_mode(new_mode)
    theme_button.configure(text=f"{current_mode}")

# Hàm kiểm tra kiểu dữ liệu của các cột
def check_column_types(df):
    global categorical_features, numeric_features
    categorical_features = []
    numeric_features = []

    for col in df.columns:
        if df[col].dtype == 'object':
            categorical_features.append(col)
        else:
            numeric_features.append(col)
    if 'median_house_value' in numeric_features:
        numeric_features.remove('median_house_value')

# Hàm phân tích dữ liệu cơ bản
def analyze_data(df):
    analysis = {}
    analysis['shape'] = df.shape
    analysis['missing_values'] = df.isna().sum().to_dict()
    analysis['dtypes'] = df.dtypes.to_dict()
    analysis['numeric_stats'] = df.describe().to_dict() if len(numeric_features) > 0 else {}
    analysis['categorical_stats'] = df[categorical_features].describe(include='object').to_dict() if len(
        categorical_features) > 0 else {}
    return analysis

def load_file():
    progress_bar.start()
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            global data
            data = pd.read_csv(file_path)

            if data.empty:
                messagebox.showerror("Lỗi", "File dữ liệu trống!")
                progress_bar.stop()
                return

            duplicates = data.duplicated().sum()
            duplicates_message = f"Đã loại bỏ {duplicates} bản ghi trùng lặp.\n\n" if duplicates > 0 else f"Không phát hiện bản ghi trùng lặp.\n\n"
            if duplicates > 0:
                data.drop_duplicates(inplace=True)

            check_column_types(data)
            data_analysis = analyze_data(data)

            textbox.delete("1.0", "end")
            textbox.insert("end", f"Đã tải: {file_path}\n")
            textbox.insert("end", f"Số dòng: {data_analysis['shape'][0]}, Số cột: {data_analysis['shape'][1]}\n")
            textbox.insert("end", f"\nKiểu dữ liệu:\n")
            for col, dtype in data_analysis['dtypes'].items():
                textbox.insert("end", f"- {col}: {dtype}\n")

            textbox.insert("end", f"\nGiá trị thiếu:\n")
            for col, count in data_analysis['missing_values'].items():
                if count > 0:
                    textbox.insert("end", f"- {col}: {count} giá trị thiếu\n")

            if sum(data_analysis['missing_values'].values()) > 0:
                textbox.insert("end", f"\nCảnh báo: Có {sum(data_analysis['missing_values'].values())} giá trị thiếu. Sẽ được xử lý tự động.\n\n")

            textbox.insert("end", duplicates_message)

            if 'ocean_proximity' in categorical_features:
                textbox.insert("end", f"\nHình 3.6. Kiểm tra dữ liệu dạng phi số\n")
                textbox.insert("end", f"print(df['ocean_proximity'])\n")
                textbox.insert("end", "# Kiểm tra dữ liệu dạng chữ\n")
                head_data = data['ocean_proximity'].head(5)
                for i, value in head_data.items():
                    textbox.insert("end", f"{i:4d}    {value}\n")
                if len(data) > 10:
                    textbox.insert("end", f"...\n")
                tail_data = data['ocean_proximity'].tail(5)
                for i, value in tail_data.items():
                    textbox.insert("end", f"{i:4d}    {value}\n")
                textbox.insert("end", f"Name: ocean_proximity, Length: {len(data)}, dtype: {data['ocean_proximity'].dtype}\n")

            if 'ocean_proximity' in categorical_features:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_data = encoder.fit_transform(data[['ocean_proximity']])
                df_encode = pd.DataFrame(encoded_data, columns=[col.split('_')[-1] for col in encoder.get_feature_names_out(['ocean_proximity'])])
                df_encode = df_encode.astype(int)

                textbox.insert("end", f"\nHình 3.7. Thực hiện chuyển đổi dữ liệu bằng One-Hot Encoding\n")
                textbox.insert("end", "# mã hóa thành category, chuyển về dạng 0-1\n")
                textbox.insert("end", f"encoded_data = encoder.fit_transform(df[['ocean_proximity']])\n")
                textbox.insert("end", f"df_encode = pd.DataFrame(encoded_data, columns=[col.split('_')[-1] for col in encoder.get_feature_names_out(['ocean_proximity'])])\n")
                textbox.insert("end", df_encode.head(4).to_string(index=True))
                textbox.insert("end", "\n...\n")
                textbox.insert("end", df_encode.tail(6).to_string(index=True))
                textbox.insert("end", f"\n{len(data)} rows x {len(df_encode.columns)} columns\n")

            outlier_messages = []
            if numeric_features:
                outlier_indices = detect_outliers_isolation_forest(data, numeric_features, contamination=0.05)
                if len(outlier_indices) > 0:
                    for col in numeric_features:
                        median_val = data[col].median()
                        data.loc[outlier_indices, col] = median_val
                        outlier_messages.append(f"Đã thay thế {len(outlier_indices)} giá trị ngoại lai trong cột {col} bằng trung vị ({median_val:.2f}).\n")
                else:
                    outlier_messages.append(f"Không phát hiện giá trị ngoại lai trong các cột số.\n")

            error_messages = []
            for col in numeric_features:
                if col in ['total_rooms', 'total_bedrooms', 'population', 'households']:
                    invalid_count = (data[col] < 0).sum()
                    if invalid_count > 0:
                        median_val = data[col].median()
                        data.loc[data[col] < 0, col] = median_val
                        error_messages.append(f"Cột {col}: Đã sửa {invalid_count} giá trị âm thành trung vị ({median_val:.2f}).\n")

            for msg in outlier_messages:
                textbox.insert("end", msg)

            for msg in error_messages:
                textbox.insert("end", msg)

            textbox.insert("end", f"\nHình 3.3. Thông tin tóm lược dữ liệu của các cột dữ liệu dạng số\n")
            textbox.insert("end", f"{'':<15} {'count':>10} {'mean':>12} {'std':>12} {'min':>12} {'25%':>12} {'50%':>12} {'75%':>12} {'max':>12}\n")
            textbox.insert("end", "-" * 95 + "\n")
            for col in numeric_features:
                stats = data_analysis['numeric_stats'].get(col, {})
                textbox.insert("end", f"{col:<15} {stats.get('count', 0):>10.0f} {stats.get('mean', 0):>12.2f} {stats.get('std', 0):>12.2f} {stats.get('min', 0):>12.2f} {stats.get('25%', 0):>12.2f} {stats.get('50%', 0):>12.2f} {stats.get('75%', 0):>12.2f} {stats.get('max', 0):>12.2f}\n")

            for checkbox in feature_checkboxes:
                checkbox.destroy()
            feature_checkboxes.clear()
            checkbox_vars.clear()

            for i, col in enumerate(numeric_features + categorical_features):
                var = tk.BooleanVar(value=False)
                checkbox = ctk.CTkCheckBox(feature_frame, text=col, variable=var, font=("Arial", 12), onvalue=True,
                                           offvalue=False)
                checkbox.grid(row=i, column=0, sticky="w", padx=10, pady=2)
                feature_checkboxes.append(checkbox)
                checkbox_vars.append(var)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không đọc được file: {str(e)}")
    progress_bar.stop()
    progress_bar.set(0)

def get_selected_features():
    return [checkbox.cget("text") for checkbox, var in zip(feature_checkboxes, checkbox_vars) if var.get()]

def create_preprocessing_pipeline(selected_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    selected_numeric = [f for f in selected_features if f in numeric_features]
    selected_categorical = [f for f in selected_features if f in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, selected_numeric),
            ('cat', categorical_transformer, selected_categorical)])
    return preprocessor

def train_model():
    progress_bar.start()
    global model, X_selected, preprocessor, selected_features
    if data is None:
        messagebox.showwarning("Thông báo", "Chưa tải dữ liệu!")
        progress_bar.stop()
        return

    selected_features = get_selected_features()
    if not selected_features:
        messagebox.showwarning("Thông báo", "Chọn ít nhất 1 đặc trưng!")
        progress_bar.stop()
        return

    if 'median_house_value' not in data.columns:
        messagebox.showerror("Lỗi", "Không tìm thấy cột 'median_house_value'!")
        progress_bar.stop()
        return

    try:
        preprocessor = create_preprocessing_pipeline(selected_features)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())])

        X = data[selected_features]
        y = data['median_house_value']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True)

        start_time = datetime.now()
        pipeline.fit(X_train, y_train)
        training_time = datetime.now() - start_time

        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        textbox.insert("end", f"\n==> Huấn luyện xong! (Thời gian: {training_time.total_seconds():.2f}s)\n")
        textbox.insert("end", f"RMSE: {rmse:.2f}\n")
        textbox.insert("end", f"MAE: {mae:.2f}\n")
        textbox.insert("end", f"R2 Score: {r2:.4f}\n")

        model = pipeline
        X_selected = X_test.copy()
        X_selected['Actual'] = y_test
        X_selected['Predicted'] = y_pred
        X_selected['Residuals'] = y_test - y_pred

        plot_results(y_test, y_pred)

    except Exception as e:
        messagebox.showerror("Lỗi", f"Lỗi khi huấn luyện mô hình: {str(e)}")

    progress_bar.stop()
    progress_bar.set(0)

def save_table_as_image(df, title, filename):
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.5))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title(title, fontsize=12, pad=20)
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()

def plot_results(y_test, y_pred):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Giá thực tế')
    plt.ylabel('Giá dự đoán')
    plt.title('Thực tế vs Dự đoán')

    residuals = y_test - y_pred
    plt.subplot(1, 3, 2)
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel('Sai số')
    plt.title('Phân bố sai số')

    plt.subplot(1, 3, 3)
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Giá dự đoán')
    plt.ylabel('Sai số')
    plt.title('Sai số vs Giá dự đoán')

    plt.tight_layout()
    plt.show()

def plot_residuals():
    if model is None or X_selected is None:
        messagebox.showwarning("Thông báo", "Chưa huấn luyện mô hình!")
        return

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_selected['Predicted'], y=X_selected['Residuals'], alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Giá dự đoán")
    plt.ylabel("Sai số (Residuals)")
    plt.title("Biểu đồ sai số")
    plt.grid(True)
    plt.show()

def save_predictions():
    if X_selected is None:
        messagebox.showwarning("Thông báo", "Chưa có kết quả dự đoán!")
        return

    save_path = filedialog.asksaveasfilename(
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")])

    if save_path:
        try:
            if save_path.endswith('.csv'):
                X_selected.to_csv(save_path, index=False)
            else:
                X_selected.to_excel(save_path, index=False)
            messagebox.showinfo("Thông báo", f"Đã lưu tại: {save_path}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu file: {str(e)}")

def plot_distribution():
    if data is None:
        messagebox.showwarning("Thông báo", "Chưa tải dữ liệu!")
        return

    selected_features = get_selected_features()
    if not selected_features:
        messagebox.showwarning("Thông báo", "Chọn đặc trưng để vẽ biểu đồ!")
        return

    for col in selected_features:
        plt.figure(figsize=(7, 5))
        if col in numeric_features:
            sns.histplot(data[col], kde=True, bins=30)
            plt.title(f"Phân phối của {col}")
        else:
            value_counts = data[col].value_counts()
            if len(value_counts) > 20:
                value_counts = value_counts[:20]
                plt.title(f"Top 20 giá trị phổ biến của {col}")
            else:
                plt.title(f"Phân phối của {col}")
            sns.barplot(x=value_counts.values, y=value_counts.index)
        plt.xlabel(col)
        plt.ylabel("Số lượng")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def important_feature():
    if model is None:
        messagebox.showwarning("Thông báo", "Chưa huấn luyện mô hình!")
        return

    try:
        selected_features = get_selected_features()
        if not selected_features:
            messagebox.showwarning("Thông báo", "Chưa chọn đặc trưng!")
            return

        feature_names = []
        if 'num' in model.named_steps['preprocessor'].named_transformers_:
            num_features = [f for f in selected_features if f in numeric_features]
            if num_features:
                feature_names.extend(num_features)

        if 'cat' in model.named_steps['preprocessor'].named_transformers_:
            selected_categorical = [f for f in selected_features if f in categorical_features]
            if selected_categorical and hasattr(model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'], 'get_feature_names_out'):
                cat_features = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(selected_categorical)
                feature_names.extend(cat_features)

        if not feature_names:
            messagebox.showwarning("Thông báo", "Không có đặc trưng nào được xử lý để tính độ quan trọng!")
            return

        coefs = model.named_steps['regressor'].coef_
        if len(feature_names) != len(coefs):
            raise ValueError(f"Số lượng đặc trưng ({len(feature_names)}) không khớp với số lượng hệ số ({len(coefs)}).")

        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefs,
            'Absolute_Coefficient': np.abs(coefs)
        }).sort_values('Absolute_Coefficient', ascending=False)

        textbox.insert("end", "\nĐộ quan trọng của các đặc trưng:\n")
        textbox.insert("end", feature_importance.to_string(index=False))
        textbox.insert("end", "\n")

        plt.figure(figsize=(10, 6))
        top_features = feature_importance.head(10)
        sns.barplot(x='Absolute_Coefficient', y='Feature', data=top_features, palette='viridis')
        plt.title('Top 10 đặc trưng quan trọng nhất')
        plt.xlabel('Giá trị tuyệt đối của hệ số')
        plt.ylabel('Đặc trưng')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể lấy độ quan trọng đặc trưng: {str(e)}")

def predict_new():
    if model is None:
        messagebox.showwarning("Thông báo", "Chưa huấn luyện mô hình!")
        return

    if not selected_features:
        messagebox.showwarning("Thông báo", "Chưa chọn đặc trưng!")
        return

    predict_window = ctk.CTkToplevel(app)
    predict_window.title("Dự đoán giá nhà - Nhập thông tin")
    predict_window.geometry("900x800")
    predict_window.resizable(True, True)
    predict_window.transient(app)
    predict_window.grab_set()

    input_data_list = []
    entries_list = []
    combobox_list = []

    main_container = ctk.CTkFrame(predict_window)
    main_container.pack(fill="both", expand=True, padx=10, pady=10)

    input_frame = ctk.CTkScrollableFrame(main_container, width=850, height=400)
    input_frame.pack(fill="both", expand=True, padx=5, pady=5)

    control_frame = ctk.CTkFrame(main_container)
    control_frame.pack(fill="x", padx=5, pady=5)

    result_frame = ctk.CTkFrame(main_container)
    result_frame.pack(fill="both", expand=True, padx=5, pady=5)

    features_info_frame = ctk.CTkFrame(result_frame)
    features_info_frame.pack(fill="x", padx=5, pady=5)

    result_text = ctk.CTkTextbox(result_frame, wrap="word", font=("Arial", 12))
    result_text.pack(fill="both", expand=True, padx=5, pady=5)

    info_label = ctk.CTkLabel(features_info_frame,
                              text="Thông tin đặc trưng đã chọn:",
                              font=("Arial", 12, "bold"))
    info_label.pack(anchor="w")

    info_text = ctk.CTkTextbox(features_info_frame, height=100, wrap="none", font=("Arial", 11))
    info_text.pack(fill="x", padx=5, pady=5)

    info_lines = []
    for feature in selected_features:
        if feature in numeric_features:
            stats = data[feature].describe()
            info_lines.append(
                f"- {feature} (numeric): Min={stats['min']:.2f}, Max={stats['max']:.2f}, Mean={stats['mean']:.2f}")
        else:
            unique_values = data[feature].nunique()
            top_value = data[feature].mode()[0]
            info_lines.append(f"- {feature} (categorical): {unique_values} giá trị, Phổ biến nhất: '{top_value}'")

    info_text.insert("1.0", "\n".join(info_lines))
    info_text.configure(state="disabled")

    def add_input_row():
        row_entries = []
        row_comboboxes = []
        row_frame = ctk.CTkFrame(input_frame)
        row_frame.pack(fill="x", pady=2)

        row_label = ctk.CTkLabel(row_frame, text=f"Dữ liệu {len(entries_list) + 1}:", width=80)
        row_label.grid(row=0, column=0, padx=2)

        for col, feature in enumerate(selected_features):
            feature_frame = ctk.CTkFrame(row_frame)
            feature_frame.grid(row=0, column=col + 1, padx=2)

            ctk.CTkLabel(feature_frame, text=feature, font=("Arial", 11)).pack()

            if feature in numeric_features:
                entry = ctk.CTkEntry(feature_frame, width=120, font=("Arial", 11))
                entry.pack()
                if not data[feature].isnull().all():
                    median_val = data[feature].median()
                    entry.insert(0, f"{median_val:.2f}")
                row_entries.append(entry)
                row_comboboxes.append(None)
            else:
                unique_values = data[feature].dropna().unique()
                if len(unique_values) > 20:
                    unique_values = np.append(unique_values[:20], "...")
                combobox = ctk.CTkComboBox(feature_frame,
                                           values=unique_values,
                                           width=120,
                                           font=("Arial", 11))
                combobox.pack()
                combobox.set("Chọn giá trị")
                row_entries.append(None)
                row_comboboxes.append(combobox)

        def remove_row(frame, idx):
            frame.destroy()
            entries_list.pop(idx)
            combobox_list.pop(idx)
            for i, (f, _, _) in enumerate(entries_list):
                f.children["!label"].configure(text=f"Dữ liệu {i + 1}:")

        remove_btn = ctk.CTkButton(row_frame, text="X",
                                   width=30,
                                   fg_color="red",
                                   hover_color="darkred",
                                   command=lambda f=row_frame, i=len(entries_list): remove_row(f, i))
        remove_btn.grid(row=0, column=len(selected_features) + 1, padx=2)

        entries_list.append((row_frame, row_entries, row_comboboxes))
        combobox_list.append(row_comboboxes)

    def remove_input_row():
        if entries_list:
            frame, _, _ = entries_list.pop()
            frame.destroy()
            combobox_list.pop()

    def clear_all_rows():
        while entries_list:
            remove_input_row()

    ctk.CTkButton(control_frame, text="Thêm bộ dữ liệu",
                  command=add_input_row,
                  fg_color="#1f6aa5",
                  hover_color="#144870").pack(side="left", padx=5)

    ctk.CTkButton(control_frame, text="Xóa bộ dữ liệu",
                  command=remove_input_row,
                  fg_color="#d35e60",
                  hover_color="#a83c3e").pack(side="left", padx=5)

    ctk.CTkButton(control_frame, text="Xóa tất cả",
                  command=clear_all_rows,
                  fg_color="#d35e60",
                  hover_color="#a83c3e").pack(side="left", padx=5)

    def perform_prediction():
        input_data_list.clear()
        result_text.configure(state="normal")
        result_text.delete("1.0", "end")

        if not entries_list:
            messagebox.showwarning("Thông báo", "Chưa nhập dữ liệu!")
            return

        for idx, (_, entries, comboboxes) in enumerate(entries_list):
            row_data = {}
            try:
                for i, feature in enumerate(selected_features):
                    if feature in numeric_features:
                        value = entries[i].get()
                        if not value:
                            messagebox.showerror("Lỗi", f"Dòng {idx + 1}: Giá trị của {feature} không được để trống!")
                            return
                        row_data[feature] = float(value)
                    else:
                        value = comboboxes[i].get()
                        if value == "Chọn giá trị":
                            messagebox.showerror("Lỗi", f"Dòng {idx + 1}: Chưa chọn giá trị cho {feature}!")
                            return
                        row_data[feature] = value

                input_data_list.append(row_data)
            except ValueError as e:
                messagebox.showerror("Lỗi", f"Dòng {idx + 1}: Giá trị không hợp lệ - {str(e)}")
                return

        input_df = pd.DataFrame(input_data_list)

        try:
            predictions = model.predict(input_df)
            result_text.insert("end", "KẾT QUẢ DỰ ĐOÁN:\n\n")
            for i, (input_data, pred) in enumerate(zip(input_data_list, predictions)):
                result_text.insert("end", f"=== DỰ ĐOÁN {i + 1} ===\n")
                result_text.insert("end", "Thông tin nhà:\n")
                for feature, value in input_data.items():
                    if feature in numeric_features:
                        result_text.insert("end", f"- {feature}: {float(value):.2f}\n")
                    else:
                        result_text.insert("end", f"- {feature}: {value}\n")
                result_text.insert("end", f"\n=> GIÁ NHÀ DỰ ĐOÁN: {pred:,.2f} USD\n\n")
            result_text.insert("end", f"\nThời gian dự đoán: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi dự đoán: {str(e)}")
        finally:
            result_text.configure(state="disabled")

    predict_btn = ctk.CTkButton(control_frame,
                                text="Dự đoán",
                                command=perform_prediction,
                                fg_color="#2aa44f",
                                hover_color="#1d7a3a")
    predict_btn.pack(side="right", padx=5)

    def export_results(predictions=None):
        if not input_data_list or 'predictions' not in locals():
            messagebox.showwarning("Thông báo", "Chưa có kết quả để xuất!")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")])

        if save_path:
            try:
                export_df = pd.DataFrame(input_data_list)
                export_df['Predicted_Price'] = predictions

                if save_path.endswith('.csv'):
                    export_df.to_csv(save_path, index=False)
                else:
                    export_df.to_excel(save_path, index=False)

                messagebox.showinfo("Thông báo", f"Đã lưu kết quả tại:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu file: {str(e)}")

    export_btn = ctk.CTkButton(control_frame,
                               text="Xuất kết quả",
                               command=export_results,
                               fg_color="#1f6aa5",
                               hover_color="#144870")
    export_btn.pack(side="right", padx=5)

    add_input_row()
    if entries_list and entries_list[0][1] and entries_list[0][1][0]:
        entries_list[0][1][0].focus()

def advanced_data_quality_check(df):
    quality_report = {}
    quality_report['duplicates'] = df.duplicated().sum()
    outliers = {}
    for col in numeric_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
    quality_report['outliers'] = outliers
    skewness = {col: stats.skew(df[col].dropna()) for col in numeric_features}
    quality_report['skewness'] = skewness
    correlation = df[numeric_features].corr().to_dict()
    quality_report['correlation'] = correlation
    return quality_report

def detect_outliers_isolation_forest(df, numeric_cols, contamination=0.05):
    if not df.empty and numeric_cols:
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(df[numeric_cols])
        return df.index[outliers == -1]
    return []

# Hàm cập nhật: Phân tích mô tả với hiển thị biểu đồ
def descriptive_analysis():
    if data is None:
        messagebox.showwarning("Thông báo", "Chưa tải dữ liệu!")
        return

    try:
        # Tạo cửa sổ mới để hiển thị các biểu đồ
        analysis_window = ctk.CTkToplevel(app)
        analysis_window.title("Phân tích mô tả")
        analysis_window.geometry("1200x800")
        analysis_window.resizable(True, True)
        analysis_window.transient(app)
        analysis_window.grab_set()

        # Khung chính chứa các biểu đồ
        main_container = ctk.CTkFrame(analysis_window)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Khung chứa các biểu đồ (scrollable)
        chart_frame = ctk.CTkScrollableFrame(main_container, width=1150, height=600)
        chart_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Textbox hiển thị giải thích và bảng tần suất
        result_frame = ctk.CTkFrame(main_container)
        result_frame.pack(fill="both", expand=True, padx=5, pady=5)
        result_text = ctk.CTkTextbox(result_frame, wrap="word", font=("Arial", 12), height=150)
        result_text.pack(fill="both", expand=True, padx=5, pady=5)

        result_text.insert("end", "=== Phân tích mô tả ===\n\n")

        # 1. Histogram của median_house_value
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.hist(data['median_house_value'], bins=10, color='skyblue', edgecolor='black')
        ax1.set_title('Hình 3.8: Histogram của median_house_value')
        ax1.set_xlabel('Giá nhà (USD)')
        ax1.set_ylabel('Số lượng')
        ax1.grid(True)
        canvas1 = FigureCanvasTkAgg(fig1, master=chart_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(pady=5)
        result_text.insert("end", ": Histogram của median_house_value\n")
        result_text.insert("end", "Phân bố giá nhà trung bình, lệch phải với ngoại lai trên 400,000 USD.\n")
        result_text.insert("end", "Ý nghĩa: Xác định xu hướng phân tán và phát hiện ngoại lai.\n\n")

        # 2. Boxplot của median_house_value
        fig2, ax2 = plt.subplots(figsize=(6, 2))
        ax2.boxplot(data['median_house_value'], vert=False)
        ax2.set_title(': Boxplot của median_house_value')
        ax2.set_xlabel('Giá nhà (USD)')
        ax2.grid(True)
        canvas2 = FigureCanvasTkAgg(fig2, master=chart_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(pady=5)
        result_text.insert("end", ": Boxplot của median_house_value\n")
        result_text.insert("end", "Hiển thị trung vị (≈180,000 USD), tứ phân vị và ngoại lai trên 500,000 USD.\n")
        result_text.insert("end", "Ý nghĩa: Cung cấp cái nhìn tổng quan về phân bố và giá trị bất thường.\n\n")

        # 3. Bar chart của ocean_proximity
        if 'ocean_proximity' in data.columns:
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.countplot(x='ocean_proximity', data=data, palette='Blues', ax=ax3)
            ax3.set_title(': Tần suất của ocean_proximity')
            ax3.set_xlabel('Danh mục')
            ax3.set_ylabel('Số lượng')
            ax3.tick_params(axis='x', rotation=45)
            canvas3 = FigureCanvasTkAgg(fig3, master=chart_frame)
            canvas3.draw()
            canvas3.get_tk_widget().pack(pady=5)
            result_text.insert("end", ": Biểu đồ cột của ocean_proximity\n")
            result_text.insert("end", "Tần suất các danh mục vị trí, với <1H OCEAN chiếm ưu thế (≈45%).\n")
            result_text.insert("end", "Ý nghĩa: Phản ánh ảnh hưởng của vị trí đến giá nhà.\n\n")
        else:
            result_text.insert("end", "Cảnh báo: Không tìm thấy cột 'ocean_proximity' để vẽ biểu đồ.\n\n")

        # 4. Scatter plot của median_income vs median_house_value với đường hồi quy
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.scatter(data['median_income'], data['median_house_value'], alpha=0.5, color='green', label='Dữ liệu')
        slope, intercept, r_value, p_value, std_err = stats.linregress(data['median_income'], data['median_house_value'])
        line = slope * data['median_income'] + intercept
        ax4.plot(data['median_income'], line, color='red', label=f'Regression line (R² = {r_value**2:.2f})')
        ax4.set_title('Hình 3.11: Scatter plot giữa median_income và median_house_value')
        ax4.set_xlabel('Thu nhập trung bình (median_income)')
        ax4.set_ylabel('Giá nhà (USD)')
        ax4.grid(True)
        ax4.legend()
        canvas4 = FigureCanvasTkAgg(fig4, master=chart_frame)
        canvas4.draw()
        canvas4.get_tk_widget().pack(pady=5)
        result_text.insert("end", ": Scatter plot giữa median_income và median_house_value\n")
        result_text.insert("end", f"Mối quan hệ tuyến tính tích cực, R² ≈ {r_value**2:.2f}.\n")
        result_text.insert("end", "Ý nghĩa: median_income là yếu tố quan trọng dự báo giá nhà.\n\n")

        # 5. Frequency table giữa ocean_proximity và median_house_value
        if 'ocean_proximity' in data.columns:
            bins = [0, 100000, 200000, 300000, 400000, 500000]
            labels = ['0-100k', '100k-200k', '200k-300k', '300k-400k', '400k-500k']
            data['price_range'] = pd.cut(data['median_house_value'], bins=bins, labels=labels, include_lowest=True)
            frequency_table = pd.crosstab(data['ocean_proximity'], data['price_range'])
            fig5, ax5 = plt.subplots(figsize=(6, 2))
            ax5.axis('tight')
            ax5.axis('off')
            table = ax5.table(cellText=frequency_table.values, colLabels=frequency_table.columns, rowLabels=frequency_table.index, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.2)
            ax5.set_title('Bảng 3.1: Tần suất giữa ocean_proximity và median_house_value')
            canvas5 = FigureCanvasTkAgg(fig5, master=chart_frame)
            canvas5.draw()
            canvas5.get_tk_widget().pack(pady=5)
            result_text.insert("end", "Bảng 3.1: Tần suất giữa ocean_proximity và median_house_value\n")
            result_text.insert("end", frequency_table.to_string())
            result_text.insert("end", "\nÝ nghĩa: Khu vực gần biển có giá cao hơn khu vực nội địa.\n\n")
        else:
            result_text.insert("end", "Cảnh báo: Không tìm thấy cột 'ocean_proximity' để tạo bảng tần suất.\n\n")

        # 6. Heatmap của correlation
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        correlation_matrix = data[['median_house_value', 'median_income', 'housing_median_age']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax6)
        ax6.set_title(' Heatmap hệ số tương quan')
        canvas6 = FigureCanvasTkAgg(fig6, master=chart_frame)
        canvas6.draw()
        canvas6.get_tk_widget().pack(pady=5)
        result_text.insert("end", " Heatmap hệ số tương quan\n")
        result_text.insert("end", "Tương quan mạnh giữa median_house_value và median_income (≈0.7).\n")
        result_text.insert("end", "Ý nghĩa: Giúp lựa chọn biến quan trọng cho mô hình.\n\n")

        result_text.insert("end", "Nhận xét:\n")
        result_text.insert("end", "- median_house_value lệch phải, có ngoại lai cần xử lý.\n")
        result_text.insert("end", "- Vị trí địa lý ảnh hưởng lớn đến giá nhà, cần mã hóa biến ocean_proximity.\n")
        result_text.insert("end", "- median_income có tương quan mạnh với giá nhà, là biến quan trọng.\n")

    except Exception as e:
        messagebox.showerror("Lỗi", f"Lỗi khi thực hiện phân tích mô tả: {str(e)}")

main_frame = ctk.CTkFrame(app, corner_radius=10)
main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
app.grid_rowconfigure(0, weight=1)
app.grid_columnconfigure(0, weight=1)

title_label = ctk.CTkLabel(main_frame, text="Phân tích giá nhà California", font=("Arial", 20, "bold"))
title_label.grid(row=0, column=0, columnspan=3, pady=10)

theme_button = ctk.CTkButton(main_frame, text="Dark", command=toggle_theme, width=120)
theme_button.grid(row=0, column=2, sticky="ne", padx=10)

progress_bar = ctk.CTkProgressBar(main_frame, width=300)
progress_bar.grid(row=1, column=0, columnspan=3, pady=5)
progress_bar.set(0)

file_frame = ctk.CTkFrame(main_frame, corner_radius=10)
file_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
load_button = ctk.CTkButton(file_frame, text="Chọn file CSV",
                            command=lambda: threading.Thread(target=load_file).start(),
                            fg_color="#1f6aa5", hover_color="#144870")
load_button.grid(row=0, column=0, padx=10, pady=10)

feature_label = ctk.CTkLabel(main_frame, text="Chọn đặc trưng (numeric):", font=("Arial", 14))
feature_label.grid(row=3, column=0, columnspan=3, pady=5)
feature_frame = ctk.CTkScrollableFrame(main_frame, width=300, height=150, corner_radius=10)
feature_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=5, sticky="ew")

button_frame = ctk.CTkFrame(main_frame, corner_radius=10)
button_frame.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
train_button = ctk.CTkButton(button_frame, text="Huấn luyện mô hình",
                             command=lambda: threading.Thread(target=train_model).start(),
                             fg_color="#1f6aa5", hover_color="#144870")
train_button.grid(row=0, column=0, padx=5, pady=5)
residuals_button = ctk.CTkButton(button_frame, text="Biểu đồ sai số", command=plot_residuals,
                                 fg_color="#1f6aa5", hover_color="#144870")
residuals_button.grid(row=0, column=1, padx=5, pady=5)
save_button = ctk.CTkButton(button_frame, text="Lưu kết quả", command=save_predictions,
                            fg_color="#1f6aa5", hover_color="#144870")
save_button.grid(row=0, column=2, padx=5, pady=5)
distribution_button = ctk.CTkButton(button_frame, text="Phân phối đặc trưng", command=plot_distribution,
                                    fg_color="#1f6aa5", hover_color="#144870")
distribution_button.grid(row=1, column=0, padx=5, pady=5)
important_button = ctk.CTkButton(button_frame, text="Đặc trưng quan trọng", command=important_feature,
                                 fg_color="#1f6aa5", hover_color="#144870")
important_button.grid(row=1, column=1, padx=5, pady=5)
predict_button = ctk.CTkButton(button_frame, text="Dự đoán giá mới", command=predict_new,
                               fg_color="#1f6aa5", hover_color="#144870")
predict_button.grid(row=1, column=2, padx=5, pady=5)

descriptive_button = ctk.CTkButton(button_frame, text="Phân tích mô tả", command=descriptive_analysis,
                                   fg_color="#1f6aa5", hover_color="#144870")
descriptive_button.grid(row=0, column=3, padx=5, pady=5)

textbox = ctk.CTkTextbox(main_frame, height=200, font=("Arial", 12))
textbox.grid(row=6, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
main_frame.grid_rowconfigure(6, weight=1)
main_frame.grid_columnconfigure(0, weight=1)

app.mainloop()
