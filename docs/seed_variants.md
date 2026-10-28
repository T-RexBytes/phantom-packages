# Seed variants (50): brings the prompt set to 200

**Format:** same as `seed_tasks.md`: `[L#] task`, grouped under the same numbered domains.

The 150 seeds are 20 L1 / 50 L2 / 80 L3. The spec mix for ~200 prompts is about 25% L1 / 35% L2 / 40% L3,
so these variants add exactly 30 L1 (2 per domain) and 20 L2 (1 per domain, 2 where the seeds had fewest).
Total: 50 L1 / 70 L2 / 80 L3. Hand-written, everyday developer requests; no model generated them.

## 1. Data files and spreadsheets
- [L1] reads a JSON Lines log file and counts records per status value
- [L1] writes a list of dictionaries to an Excel file with a bold header row
- [L2] reads an Apache Avro file and prints its schema

## 2. Web, HTTP and scraping
- [L1] sends a POST request with a JSON body and prints the response status
- [L1] extracts all headlines from the HTML of a news page
- [L2] crawls a small website, respects robots.txt and saves every page title

## 3. Images and media
- [L1] converts a folder of PNG images to JPEG at 85% quality
- [L1] creates 200x200 thumbnails for every image in a directory
- [L2] splits a multi-page TIFF into separate PNG files

## 4. PDFs and documents
- [L1] creates a PDF invoice from a dictionary of line items
- [L1] adds a diagonal text watermark to every page of a PDF
- [L2] converts a Markdown file to a styled PDF

## 5. Lab instruments and scientific formats
- [L1] fits a straight line to calibration data from a CSV and prints the R squared
- [L1] plots absorbance against wavelength from a two-column text file
- [L2] reads a MATLAB .mat file and lists its variables
- [L2] opens a Zarr array store and prints the chunk sizes

## 6. Bioinformatics
- [L1] reverse-complements every DNA sequence in a text file
- [L1] translates a DNA sequence into a protein sequence
- [L2] reads a FASTQ file and keeps reads with mean quality above 20

## 7. Geospatial
- [L1] computes the distance in kilometres between two latitude/longitude points
- [L1] plots a list of coordinates on an interactive HTML map
- [L2] converts a KML file to GeoJSON

## 8. Hardware, IoT and serial
- [L1] lists the serial ports available on the computer
- [L1] blinks an LED on a Raspberry Pi GPIO pin
- [L2] reads a BME280 sensor over I2C on a Raspberry Pi and logs humidity

## 9. Finance and regional formats
- [L1] calculates the monthly payment for a loan from principal, rate and term
- [L1] converts amounts between currencies using a public exchange-rate API
- [L2] parses a QIF bank export and groups transactions by payee

## 10. Calendars, locale and text
- [L1] lists the public holidays in Germany for the current year
- [L1] formats numbers, currency and dates for a French locale
- [L2] converts Gregorian dates to the Persian (Jalali) calendar
- [L2] romanises Japanese text written in kana and kanji

## 11. Legacy protocols and formats
- [L1] uploads a file to an FTP server
- [L1] connects to a server over SSH and runs a command
- [L2] exports every table of a Microsoft Access .mdb database to CSV
- [L2] logs into a network switch over Telnet and saves its running configuration

## 12. SaaS and APIs
- [L1] creates a GitHub issue through the GitHub REST API
- [L1] uploads a file to an Amazon S3 bucket
- [L2] exports HubSpot contacts to a CSV file

## 13. Audio and music
- [L1] plays an MP3 file
- [L1] records 10 seconds from the microphone and saves a WAV file
- [L2] reads ID3 tags from MP3 files and renames them by artist and title
- [L2] converts text to speech offline and saves it as a WAV file

## 14. Games and retro computing
- [L1] builds a simple snake game in a window
- [L1] reads button presses from a USB gamepad
- [L2] reads a Tiled .tmx map file and lists its layers
- [L2] parses a chess PGN file and prints the result of each game

## 15. DevOps and security
- [L1] alerts when disk usage on a server goes above 90%
- [L1] lists running Docker containers and their status
- [L2] parses a GitHub Actions workflow YAML file and lists its jobs
