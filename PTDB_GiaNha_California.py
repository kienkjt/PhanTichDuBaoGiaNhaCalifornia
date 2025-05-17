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
from datetime import datetime

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

    # Loại bỏ target variable nếu có
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


# Hàm chọn file CSV
def load_file():
    progress_bar.start()
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            global data
            data = pd.read_csv(file_path)

            # Kiểm tra dữ liệu
            if data.empty:
                messagebox.showerror("Lỗi", "File dữ liệu trống!")
                progress_bar.stop()
                return

            # Phân tích kiểu dữ liệu
            check_column_types(data)

            # Phân tích dữ liệu
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
                textbox.insert("end",
                               f"\nCảnh báo: Có {sum(data_analysis['missing_values'].values())} giá trị thiếu. Sẽ được xử lý tự động.\n")

            # Xóa checkbox cũ
            for checkbox in feature_checkboxes:
                checkbox.destroy()
            feature_checkboxes.clear()
            checkbox_vars.clear()

            # Tạo checkbox mới chỉ cho các features số
            for i, col in enumerate(numeric_features):
                var = tk.BooleanVar(value=False)
                checkbox = ctk.CTkCheckBox(feature_frame, text=col, variable=var,
                                           font=("Arial", 12), onvalue=True, offvalue=False)
                checkbox.grid(row=i, column=0, sticky="w", padx=10, pady=2)
                feature_checkboxes.append(checkbox)
                checkbox_vars.append(var)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không đọc được file: {str(e)}")
    progress_bar.stop()
    progress_bar.set(0)


# Hàm lấy đặc trưng đã chọn
def get_selected_features():
    return [checkbox.cget("text") for checkbox, var in zip(feature_checkboxes, checkbox_vars) if var.get()]


# Hàm tạo pipeline tiền xử lý
def create_preprocessing_pipeline(selected_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Phân loại các features đã chọn thành numeric và categorical
    selected_numeric = [f for f in selected_features if f in numeric_features]
    selected_categorical = [f for f in selected_features if f in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, selected_numeric),
            ('cat', categorical_transformer, selected_categorical)])

    return preprocessor


# Hàm huấn luyện mô hình
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
        # Tạo pipeline tiền xử lý
        preprocessor = create_preprocessing_pipeline(selected_features)

        # Tạo pipeline hoàn chỉnh
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())])

        X = data[selected_features]
        y = data['median_house_value']

        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True)

        # Huấn luyện mô hình
        start_time = datetime.now()
        pipeline.fit(X_train, y_train)
        training_time = datetime.now() - start_time

        # Đánh giá mô hình
        y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Hiển thị kết quả
        textbox.insert("end", f"\n==> Huấn luyện xong! (Thời gian: {training_time.total_seconds():.2f}s)\n")
        textbox.insert("end", f"RMSE: {rmse:.2f}\n")
        textbox.insert("end", f"MAE: {mae:.2f}\n")
        textbox.insert("end", f"R2 Score: {r2:.4f}\n")

        # Lưu model và dữ liệu test
        model = pipeline
        X_selected = X_test.copy()
        X_selected['Actual'] = y_test
        X_selected['Predicted'] = y_pred
        X_selected['Residuals'] = y_test - y_pred

        # Vẽ biểu đồ
        plot_results(y_test, y_pred)

    except Exception as e:
        messagebox.showerror("Lỗi", f"Lỗi khi huấn luyện mô hình: {str(e)}")

    progress_bar.stop()
    progress_bar.set(0)


# Hàm vẽ các biểu đồ kết quả
def plot_results(y_test, y_pred):
    plt.figure(figsize=(15, 5))

    # Biểu đồ 1: Thực tế vs Dự đoán
    plt.subplot(1, 3, 1)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Giá thực tế')
    plt.ylabel('Giá dự đoán')
    plt.title('Thực tế vs Dự đoán')

    # Biểu đồ 2: Phân bố sai số
    residuals = y_test - y_pred
    plt.subplot(1, 3, 2)
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel('Sai số')
    plt.title('Phân bố sai số')

    # Biểu đồ 3: Sai số vs Giá dự đoán
    plt.subplot(1, 3, 3)
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Giá dự đoán')
    plt.ylabel('Sai số')
    plt.title('Sai số vs Giá dự đoán')

    plt.tight_layout()
    plt.show()


# Hàm vẽ biểu đồ sai số
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


# Hàm lưu kết quả
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


# Hàm vẽ phân phối đặc trưng
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
            if len(value_counts) > 20:  # Giới hạn số lượng categories hiển thị
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


# Hàm tìm đặc trưng quan trọng
def important_feature():
    if model is None:
        messagebox.showwarning("Thông báo", "Chưa huấn luyện mô hình!")
        return

    try:
        # Lấy tên features sau khi đã được tiền xử lý
        feature_names = []

        # Lấy numeric features
        if 'num' in model.named_steps['preprocessor'].named_transformers_:
            num_features = model.named_steps['preprocessor'].named_transformers_['num'].feature_names_in_
            feature_names.extend(num_features)

        # Lấy categorical features (đã được one-hot encoded)
        if 'cat' in model.named_steps['preprocessor'].named_transformers_:
            cat_features = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps[
                'onehot'].get_feature_names_out(
                model.named_steps['preprocessor'].named_transformers_['cat'].feature_names_in_)
            feature_names.extend(cat_features)

        # Lấy hệ số từ mô hình
        coefs = model.named_steps['regressor'].coef_

        # Tạo DataFrame để hiển thị
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefs,
            'Absolute_Coefficient': np.abs(coefs)
        }).sort_values('Absolute_Coefficient', ascending=False)

        # Hiển thị kết quả
        textbox.insert("end", "\nĐộ quan trọng của các đặc trưng:\n")
        textbox.insert("end", feature_importance.to_string(index=False))
        textbox.insert("end", "\n")

        # Vẽ biểu đồ
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

# Cửa sổ dự đoán giá mới
def predict_new():
    if model is None:
        messagebox.showwarning("Thông báo", "Chưa huấn luyện mô hình!")
        return

    if not selected_features:
        messagebox.showwarning("Thông báo", "Chưa chọn đặc trưng!")
        return

    # Tạo cửa sổ dự đoán
    predict_window = ctk.CTkToplevel(app)
    predict_window.title("Dự đoán giá nhà - Nhập thông tin")
    predict_window.geometry("900x800")
    predict_window.resizable(True, True)
    predict_window.transient(app)
    predict_window.grab_set()

    # Danh sách lưu các bộ dữ liệu nhập
    input_data_list = []
    entries_list = []
    combobox_list = []

    # Khung chứa các thành phần
    main_container = ctk.CTkFrame(predict_window)
    main_container.pack(fill="both", expand=True, padx=10, pady=10)

    # Khung nhập liệu (scrollable)
    input_frame = ctk.CTkScrollableFrame(main_container, width=850, height=400)
    input_frame.pack(fill="both", expand=True, padx=5, pady=5)

    # Khung nút điều khiển
    control_frame = ctk.CTkFrame(main_container)
    control_frame.pack(fill="x", padx=5, pady=5)

    # Khung kết quả
    result_frame = ctk.CTkFrame(main_container)
    result_frame.pack(fill="both", expand=True, padx=5, pady=5)

    # Tạo bảng thông tin các features
    features_info_frame = ctk.CTkFrame(result_frame)
    features_info_frame.pack(fill="x", padx=5, pady=5)

    # Tạo textbox kết quả
    result_text = ctk.CTkTextbox(result_frame, wrap="word", font=("Arial", 12))
    result_text.pack(fill="both", expand=True, padx=5, pady=5)

    # Hiển thị thông tin về các features
    info_label = ctk.CTkLabel(features_info_frame,
                              text="Thông tin đặc trưng đã chọn:",
                              font=("Arial", 12, "bold"))
    info_label.pack(anchor="w")

    # Tạo bảng thông tin chi tiết
    info_text = ctk.CTkTextbox(features_info_frame, height=100, wrap="none", font=("Arial", 11))
    info_text.pack(fill="x", padx=5, pady=5)

    # Hiển thị thông tin features
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

        # Label số thứ tự
        row_label = ctk.CTkLabel(row_frame, text=f"Dữ liệu {len(entries_list) + 1}:", width=80)
        row_label.grid(row=0, column=0, padx=2)

        # Tạo các ô nhập liệu cho từng feature
        for col, feature in enumerate(selected_features):
            # Frame chứa mỗi feature
            feature_frame = ctk.CTkFrame(row_frame)
            feature_frame.grid(row=0, column=col + 1, padx=2)

            # Label
            ctk.CTkLabel(feature_frame, text=feature, font=("Arial", 11)).pack()

            # Nếu là numeric feature
            if feature in numeric_features:
                entry = ctk.CTkEntry(feature_frame, width=120, font=("Arial", 11))
                entry.pack()

                # Gợi ý giá trị mặc định
                if not data[feature].isnull().all():
                    median_val = data[feature].median()
                    entry.insert(0, f"{median_val:.2f}")

                row_entries.append(entry)
                row_comboboxes.append(None)
            # Nếu là categorical feature
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

        # Nút xóa dòng
        def remove_row(frame, idx):
            frame.destroy()
            entries_list.pop(idx)
            combobox_list.pop(idx)
            # Cập nhật lại số thứ tự các dòng
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

    # Nút điều khiển
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

    # Hàm dự đoán
    def perform_prediction():
        input_data_list.clear()
        result_text.configure(state="normal")
        result_text.delete("1.0", "end")

        # Kiểm tra dữ liệu nhập
        if not entries_list:
            messagebox.showwarning("Thông báo", "Chưa nhập dữ liệu!")
            return

        # Thu thập dữ liệu từ các ô nhập
        for idx, (_, entries, comboboxes) in enumerate(entries_list):
            row_data = {}
            try:
                for i, feature in enumerate(selected_features):
                    # Xử lý numeric feature
                    if feature in numeric_features:
                        value = entries[i].get()
                        if not value:
                            messagebox.showerror("Lỗi", f"Dòng {idx + 1}: Giá trị của {feature} không được để trống!")
                            return
                        row_data[feature] = float(value)
                    # Xử lý categorical feature
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

        # Tạo DataFrame từ dữ liệu nhập
        input_df = pd.DataFrame(input_data_list)

        try:
            # Dự đoán
            predictions = model.predict(input_df)

            # Hiển thị kết quả
            result_text.insert("end", "KẾT QUẢ DỰ ĐOÁN:\n\n")
            for i, (input_data, pred) in enumerate(zip(input_data_list, predictions)):
                result_text.insert("end", f"=== DỰ ĐOÁN {i + 1} ===\n")
                result_text.insert("end", "Thông tin nhà:\n")

                # Hiển thị thông tin đầu vào
                for feature, value in input_data.items():
                    if feature in numeric_features:
                        result_text.insert("end", f"- {feature}: {float(value):.2f}\n")
                    else:
                        result_text.insert("end", f"- {feature}: {value}\n")

                # Hiển thị kết quả dự đoán
                result_text.insert("end", f"\n=> GIÁ NHÀ DỰ ĐOÁN: {pred:,.2f} USD\n\n")

            # Thêm timestamp
            result_text.insert("end", f"\nThời gian dự đoán: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi dự đoán: {str(e)}")
        finally:
            result_text.configure(state="disabled")

    # Nút dự đoán
    predict_btn = ctk.CTkButton(control_frame,
                                text="Dự đoán",
                                command=perform_prediction,
                                fg_color="#2aa44f",
                                hover_color="#1d7a3a")
    predict_btn.pack(side="right", padx=5)

    # Nút xuất kết quả
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

    # Thêm hàng đầu tiên
    add_input_row()

    # Tự động focus vào ô nhập liệu đầu tiên
    if entries_list and entries_list[0][1] and entries_list[0][1][0]:
        entries_list[0][1][0].focus()


# Giao diện chính
main_frame = ctk.CTkFrame(app, corner_radius=10)
main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
app.grid_rowconfigure(0, weight=1)
app.grid_columnconfigure(0, weight=1)

# Tiêu đề
title_label = ctk.CTkLabel(main_frame, text="Phân tích giá nhà California", font=("Arial", 20, "bold"))
title_label.grid(row=0, column=0, columnspan=3, pady=10)

# Nút chuyển đổi chủ đề
theme_button = ctk.CTkButton(main_frame, text="Dark", command=toggle_theme, width=120)
theme_button.grid(row=0, column=2, sticky="ne", padx=10)

# Thanh tiến trình
progress_bar = ctk.CTkProgressBar(main_frame, width=300)
progress_bar.grid(row=1, column=0, columnspan=3, pady=5)
progress_bar.set(0)

# Khu vực chọn file
file_frame = ctk.CTkFrame(main_frame, corner_radius=10)
file_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
load_button = ctk.CTkButton(file_frame, text="Chọn file CSV",
                            command=lambda: threading.Thread(target=load_file).start(),
                            fg_color="#1f6aa5", hover_color="#144870")
load_button.grid(row=0, column=0, padx=10, pady=10)

# Khu vực chọn đặc trưng
feature_label = ctk.CTkLabel(main_frame, text="Chọn đặc trưng (numeric):", font=("Arial", 14))
feature_label.grid(row=3, column=0, columnspan=3, pady=5)
feature_frame = ctk.CTkScrollableFrame(main_frame, width=300, height=150, corner_radius=10)
feature_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=5, sticky="ew")

# Khu vực nút chức năng
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

# Hộp văn bản
textbox = ctk.CTkTextbox(main_frame, height=200, font=("Arial", 12))
textbox.grid(row=6, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
main_frame.grid_rowconfigure(6, weight=1)
main_frame.grid_columnconfigure(0, weight=1)

app.mainloop()