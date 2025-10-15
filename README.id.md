# Financial Analysis Crew — Panduan Bahasa Indonesia

Selamat datang! Proyek ini memperlihatkan cara mengorkestrasikan sekelompok agen AI untuk melakukan riset sebuah perusahaan publik, menganalisis data keuangan, membaca dokumen EDGAR (SEC), dan menghasilkan laporan investasi. Proyek ini menggunakan CrewAI dengan beberapa tools untuk web/news search, perhitungan matematika sederhana, dan pengambilan dokumen SEC. Tersedia juga dashboard Streamlit opsional untuk memantau agen secara real-time.

Jika Anda baru dalam Python, panduan ini ditulis untuk Anda: langkah-langkah instalasi, konfigurasi, dan cara menjalankan aplikasi dijelaskan secara bertahap.

---

## Apa yang Anda dapatkan

- CLI assistant yang meminta ticker atau nama perusahaan, lalu menghasilkan laporan riset dengan mengorkestrasi beberapa agen spesialis.
- Streamlit dashboard (`streamlit_app.py`) untuk memantau progres agen, melihat pemanggilan tool, dan membaca rekomendasi akhir lewat browser.
- Tools yang dapat digunakan ulang untuk web/news search, kalkulator, dan pengambilan filing SEC.

---

## Prasyarat

- **Python 3.10 atau lebih baru.** Pasang dari python.org jika belum terpasang.
- **Package manager.** Proyek ini diuji dengan `uv`, tetapi `pip` biasa juga bisa digunakan. `uv` memudahkan pembuatan virtual environment untuk pemula.
- **Model Ollama atau endpoint LangChain yang kompatibel** (LLM) yang dapat diakses melalui `MODEL` dan `MODEL_BASE_URL`.
- **Kunci/API opsional tetapi direkomendasikan:**
  - `SERPER_API_KEY` untuk mengaktifkan fitur search/news via Serper.dev.
  - `EDGAR_IDENTITY` (atau `SEC_IDENTITY`/`SEC_CONTACT`) — alamat email yang diperlukan saat mengakses EDGAR.

---

## Cara cepat (direkomendasikan: uv)

```bash
# Install uv (sekali saja)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Dari root proyek
uv sync
```

`uv sync` akan membaca `pyproject.toml` dan `uv.lock`, membuat lingkungan terisolasi, dan menginstal dependensi.

### Jika menggunakan pip saja

```bash
# Buat dan aktifkan virtual environment (Linux/macOS)
python3 -m venv .venv
source .venv/bin/activate

# Install dependensi
pip install crewai langchain-ollama edgartools streamlit python-dotenv requests
```

> Tip: Jika Anda memakai Windows, jalankan `.venv\\Scripts\\activate` untuk mengaktifkan environment.

---

## Konfigurasi environment variables

Buat file `.env` di root proyek dan isi variabel yang diperlukan, contohnya:

```ini
MODEL=llama3.1:8b
MODEL_BASE_URL=http://localhost:11434
SERPER_API_KEY=your_serper_key
EDGAR_IDENTITY=you@example.com
EMBEDDING_MODEL=llama3.1:8b
STREAMLIT_SERVER_PORT=8501
```

Proyek ini otomatis memuat `.env` ketika Anda menjalankan `main.py` atau `streamlit_app.py`.

---

## Menjalankan CLI

```bash
uv run python main.py
```

Masukkan ticker (misal `AAPL`) saat diminta. Aplikasi akan membuat `FinancialCrew` dengan beberapa agen dan mencetak laporan akhir ke terminal.

---

## Menjalankan Streamlit dashboard

```bash
uv run streamlit run streamlit_app.py
```

Buka URL yang tercetak (biasanya `http://localhost:8501`) lalu masukkan ticker untuk memulai run dan melihat aktivitas agen secara real-time.

---

## Struktur proyek

```
├── main.py
├── streamlit_app.py
├── agents.py
├── tasks.py
├── listeners.py
├── tools/
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Perluasan dan tips

- Anda dapat menukar atau menambahkan tools/agent baru (misal sentiment analysis atau charting).
- Simpan output agen ke database bila ingin analisis historis.
- Deploy Streamlit ke Streamlit Community Cloud jika semua API key dikelola secara aman.

Jika Anda baru di Python: coba jalankan contoh, buka file sumber, dan ubah sedikit—itu cara terbaik belajar.

Selamat mencoba! 📈
