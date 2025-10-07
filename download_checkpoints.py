import os
import gdown

checkpoint_links = [
    "https://drive.google.com/drive/folders/1qnVdmcdcteP_FcsXmiFKuhlWWbc1xCX8",
    "https://drive.google.com/drive/folders/1DQ7SlpyjhK2RdWk7qTlnU7Y6qgcbY8WV",
    "https://drive.google.com/drive/folders/1nhlwQPRvhya7uJppqgU70zxnO5cXXAIJ",
    "https://drive.google.com/drive/folders/1CmZ4RD_SAPVnr49nsxqnqB-oO6maU8dm",
    "https://drive.google.com/drive/folders/1YWQAjZFn-sSLZs2a3xZimgPsWWzlXcon",
    "https://drive.google.com/drive/folders/1GjGp9KLHoUiJxYmHvum4kCSI7hKe794L",
    "https://drive.google.com/drive/folders/1L0Tz3tg9W7pu4SW3mfnaMt7rci8_AzNv",
    "https://drive.google.com/drive/folders/11hAaKmdSfZ7Ma2l9rCVFr_VP3yQvLC49",
    "https://drive.google.com/drive/folders/1TxuRL5T6TtcciE7Weys44KNP3XfBft1b",
    "https://drive.google.com/drive/folders/1KdiV-fGA7gT6wTzp9hpgBzD5yTbNC34e",
    "https://drive.google.com/drive/folders/1Ca6CL6H18VMWYUI6sfM6qkrkum5wtkbH",
    "https://drive.google.com/drive/folders/1Wb8XUXVGUt-AKwbGTMVxSWLIrUVpMATv",
    "https://drive.google.com/drive/folders/1O3RCa9qzBEYgyOsaxCfgr-Hxu-k-MFUq",
    "https://drive.google.com/drive/folders/1drT4qf6nhH9jfoZsZL6ZYJ98mBE1Uz8Y",
    "https://drive.google.com/drive/folders/1ox12R--XJOoEFWZCd8G9RKJZU_tWUml9",
    "https://drive.google.com/drive/folders/1aD0OG0o9EGkOIqtiwvB0IadooPsgeeJh",
    "https://drive.google.com/drive/folders/1RzxgPEqVQKN3-zXsOPARgEwNIcaWEXSZ",
    "https://drive.google.com/drive/folders/1bm4M8-lfFvDaGBo_6EMD5uMTZvSrgilT",
    "https://drive.google.com/drive/folders/18kLS66XvgL7-ftFS5GFsDtynr42y9-UJ",
    "https://drive.google.com/drive/folders/18WkEpxUOfw-q4n6wQ6hE3cr6Y-q77OGS",
    "https://drive.google.com/drive/folders/1wLfogcriggJoD9kWGT1gvuWQ6Sb1_A4s",
    "https://drive.google.com/drive/folders/1xmNR-dbsb2aPdVWMn4LhkktNeWU6L47S",
    "https://drive.google.com/drive/folders/1x5CmDSALerc_fodK9ADpehacDkjG4S-h",
    "https://drive.google.com/drive/folders/1hgJ-B787VEcfIAVBj0zAHukbIqbgrW-W",
    "https://drive.google.com/drive/folders/1WlEZynPzLFfva6BKESIT-_tv53uravLG",
    "https://drive.google.com/drive/folders/1ZEgOEX7cES3pzVQn0ryc3h7yEWffEZJH",
    "https://drive.google.com/drive/folders/1VxOAGlJQxlb2XPEG5S8AlyDWPL7lBWbC",
    "https://drive.google.com/drive/folders/1S0syfZDmx3uD09b6Bu9dKr0SOeecw7_b",
    "https://drive.google.com/drive/folders/1350F1giUy9cKzUAjYgsQrneS5ijV_M5P",
    "https://drive.google.com/drive/folders/11RG-y5FmZUU4JwCWJ3e3E1mRwCdTbzMR",
    "https://drive.google.com/drive/folders/1y7s6UfvfNceFs9op5z98IEtRDh1Xg131",
    "https://drive.google.com/drive/folders/1UQKqYV2_Is5mwntCpXY6xkbiXyNerzX6",
    "https://drive.google.com/drive/folders/14OEEAQDB9h6ufch_dGaOzdzqFsp3gSjI",
    "https://drive.google.com/drive/folders/1xuydMkVmfGNMzmPpXBecNazDBC7rUMI_",
    "https://drive.google.com/drive/folders/1fgk0mUpn77cuOZFyaEtOrCGhJsXWgJO1",
]

os.makedirs("./checkpoints", exist_ok=True)
gdown.download("https://drive.google.com/file/d/1a8zVYZ7-cLAmTn512XLwFu23YvloK67q", "checkpoints/final_stacking_model.pkl", quiet=False, use_cookies=True, fuzzy=True)
gdown.download("https://drive.google.com/file/d/1H4YV4jSt53IdE6UaE52Q5Tjej2dae7Rn", "checkpoints/optimal_weights.pkl", quiet=False, use_cookies=True, fuzzy=True)

for id, link in enumerate(checkpoint_links):
    os.makedirs(f"./checkpoints/{id}", exist_ok=True)
    gdown.download_folder(link, output=f"./checkpoints/{id}", quiet=False, use_cookies=True)