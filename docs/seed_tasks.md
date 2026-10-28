# Seed tasks (150): 15 domains × 10

**Format:** `[L#] task`

- **L1** = common, **L2** = specialised, **L3** = obscure. Ghosts show up mostly at L3.
- Each task fills the template: *"Write a short Python script that {task}. Begin with the exact `pip install` command it needs."*
- Expand to about 200 with variants: different file formats, regions or devices.

## 1. Data files and spreadsheets
- [L1] reads a CSV of sales and prints total revenue per month
- [L1] converts an Excel .xlsx workbook with three sheets into separate CSV files
- [L1] merges two JSON files of users by email and writes the result
- [L2] reads a Parquet file and writes a summary of null counts per column
- [L2] reads an ODS (LibreOffice Calc) spreadsheet and lists formulas in each cell
- [L2] converts a SAS .sas7bdat file to CSV
- [L2] reads a SPSS .sav survey file and prints the variable labels
- [L3] reads a Stata .dta file from version 8 and preserves value labels
- [L3] parses a fixed-width mainframe report with a COBOL copybook layout
- [L3] reads a dBase .dbf file with Cyrillic CP866 encoding

## 2. Web, HTTP and scraping
- [L1] downloads a web page and lists all links on it
- [L1] calls a REST API with retries and exponential backoff
- [L1] scrapes a table from a Wikipedia page into a DataFrame
- [L2] renders a JavaScript-heavy page headlessly and saves a screenshot
- [L2] parses a sitemap.xml index and checks every URL for a 404
- [L2] reads a WARC web-archive file and extracts the HTML responses
- [L2] fetches an RSS and an Atom feed and merges them by date
- [L3] queries a Gopher server and prints the menu
- [L3] parses a HAR file and reports the slowest third-party requests
- [L3] reads Memento TimeMap data for a URL and lists archived snapshots

## 3. Images and media
- [L1] resizes all JPEGs in a folder to 800px wide
- [L1] reads EXIF data from photos and sorts them by date taken
- [L2] converts HEIC iPhone photos to PNG
- [L2] detects faces in an image and blurs them
- [L2] extracts frames from a video every 5 seconds
- [L2] reads a DICOM medical image and saves a PNG preview
- [L3] reads a FITS astronomy image and applies a zscale stretch
- [L3] converts a Netpbm PAM file with alpha to WebP
- [L3] reads XMP sidecar files from Lightroom and updates star ratings
- [L3] decodes a JPEG XL image and reports its colour profile

## 4. PDFs and documents
- [L1] extracts text from every page of a PDF
- [L1] merges several PDFs into one
- [L2] extracts tables from a scanned PDF invoice
- [L2] converts a .docx file to Markdown keeping headings
- [L2] fills a PDF form (AcroForm) from a CSV row
- [L2] reads an EPUB and prints the table of contents
- [L3] extracts embedded files from a PDF/A-3 (ZUGFeRD) e-invoice
- [L3] parses a legacy WordPerfect .wpd file into text
- [L3] reads a DjVu document and extracts the hidden text layer
- [L3] converts a LaTeX .bib file into CSL-JSON

## 5. Lab instruments and scientific formats
- [L2] reads an HDF5 file and lists all datasets with shapes
- [L2] reads a NetCDF climate file and plots monthly temperature
- [L3] reads a Thermo .raw mass-spectrometry file and prints the top 10 peaks
- [L3] parses an mzML file and computes total ion current
- [L3] reads a Bruker NMR fid file and Fourier transforms it
- [L3] reads Gaussian .log output and extracts the final SCF energy
- [L3] parses an Agilent ChemStation .ch chromatogram
- [L3] reads a Zeiss .czi microscopy image and saves the max projection
- [L3] reads an ABF electrophysiology recording and detects spikes
- [L3] parses a JCAMP-DX infrared spectrum

## 6. Bioinformatics
- [L1] counts GC content of sequences in a FASTA file
- [L2] reads a VCF file and lists variants with quality above 30
- [L2] parses a GFF3 annotation and extracts all exon coordinates
- [L2] reads a BAM file and computes coverage for one region
- [L2] runs a BLAST search against NCBI for a sequence
- [L3] reads a PDB protein structure and computes the radius of gyration
- [L3] parses a Newick phylogenetic tree and prints the leaves
- [L3] reads a .h5ad single-cell file and lists cell types
- [L3] converts a GenBank flat file to a feature table
- [L3] reads Oxford Nanopore POD5 signal data for one read

## 7. Geospatial
- [L1] reads a GeoJSON file and prints the area of each polygon
- [L2] reads a shapefile and reprojects it to EPSG:4326
- [L2] reads a GPX track and computes total elevation gain
- [L2] reads a GeoTIFF and samples the value at a coordinate
- [L2] geocodes a list of addresses and saves latitude/longitude
- [L3] reads a KMZ file with nested folders and flattens placemarks
- [L3] converts OSM PBF data for a city into a road graph
- [L3] reads a LAS/LAZ lidar point cloud and computes a canopy height model
- [L3] parses NMEA sentences from a GPS log file
- [L3] reads an S-57 nautical chart file and lists buoys

## 8. Hardware, IoT and serial
- [L1] reads lines from a serial port at 9600 baud
- [L2] reads temperature from a DS18B20 sensor on a Raspberry Pi
- [L2] talks Modbus RTU to an energy meter and reads voltage
- [L2] subscribes to an MQTT topic and logs messages to SQLite
- [L2] reads BLE advertisements and prints device names
- [L3] reads a CAN bus OBD-II PID for engine RPM via a USB adapter
- [L3] controls a DMX512 light fixture over an Enttec USB interface
- [L3] reads data from a Garmin .FIT activity file
- [L3] talks to a Zigbee coordinator and lists paired devices
- [L3] reads a LoRaWAN uplink payload and decodes Cayenne LPP

## 9. Finance and regional formats
- [L1] downloads daily stock prices and computes a 20-day moving average
- [L2] parses an OFX bank statement
- [L2] validates an IBAN and extracts the bank code
- [L2] generates a SEPA credit-transfer XML (pain.001) from a CSV
- [L3] parses an Indian GSTR-2B JSON and reconciles it with purchase invoices
- [L3] reads a Brazilian NF-e XML invoice and totals the taxes
- [L3] parses a SWIFT MT940 statement
- [L3] generates a Swiss QR-bill PDF
- [L3] reads a Japanese Zengin bank transfer file
- [L3] validates a Mexican CFDI 4.0 XML invoice

## 10. Calendars, locale and text
- [L1] converts timestamps between time zones for a meeting schedule
- [L2] transliterates Hindi text in Devanagari to Latin script
- [L2] detects the language of each line in a text file
- [L3] converts Gregorian dates to the Vietnamese lunar calendar
- [L3] converts dates to the Nepali Bikram Sambat calendar
- [L3] converts dates to the Hijri (Umm al-Qura) calendar
- [L3] computes Hebrew calendar holidays for a year
- [L3] converts Ethiopian calendar dates to Gregorian
- [L3] segments Thai text into words
- [L3] converts traditional Chinese characters to simplified with regional variants

## 11. Legacy protocols and formats
- [L2] parses a mbox mailbox and counts emails per sender
- [L2] reads a .pst Outlook archive and exports messages
- [L3] decodes EBCDIC text from a mainframe dump
- [L3] reads a COBOL copybook and decodes COMP-3 packed decimals
- [L3] parses FidoNet packet (.pkt) files
- [L3] reads an old Lotus 1-2-3 .wk1 spreadsheet
- [L3] parses an X12 EDI 850 purchase order
- [L3] parses a HL7 v2 ADT message into JSON
- [L3] reads an IBM AS/400 save file
- [L3] parses a UUCP-era Usenet news spool

## 12. SaaS and APIs
- [L1] posts a message to a Slack channel via webhook
- [L1] reads rows from a Google Sheet
- [L2] creates a Notion database page from a dict
- [L2] lists Jira issues assigned to me in the current sprint
- [L2] syncs Airtable records to a Postgres table
- [L2] downloads all attachments from a Zoho Mail folder
- [L3] pulls invoices from Tally ERP via its XML interface
- [L3] reads leads from a Bitrix24 CRM
- [L3] exports Odoo sales orders through XML-RPC
- [L3] pulls shipment status from the Delhivery API

## 13. Audio and music
- [L1] converts WAV files to MP3
- [L2] detects BPM of a song
- [L2] reads a MIDI file and prints the notes per track
- [L2] transcribes a speech recording to text offline
- [L3] parses an ABC notation tune and renders it to MIDI
- [L3] reads a ProTracker .mod file and lists samples
- [L3] reads a MusicXML score and transposes it up a fifth
- [L3] reads a Sonic Visualiser .sv annotation file
- [L3] converts a SoundFont .sf2 instrument to WAV samples
- [L3] decodes an Ambisonic B-format recording to stereo

## 14. Games and retro computing
- [L2] reads a Minecraft region file and lists chunks
- [L2] parses a Steam VDF (KeyValues) config
- [L3] reads a Game Boy ROM header and validates the checksum
- [L3] lists lumps in a DOOM WAD file
- [L3] runs a Z-machine story file and prints the opening text
- [L3] reads a Commodore 64 .d64 disk image directory
- [L3] parses an NES .nes iNES header
- [L3] reads an Amiga ADF disk image
- [L3] decodes a PICO-8 .p8.png cartridge
- [L3] parses an old Quake .bsp map and counts entities

## 15. DevOps and security
- [L1] parses nginx access logs and lists the top 10 IPs
- [L1] checks when a website's TLS certificate expires
- [L2] reads a CycloneDX SBOM and lists components with licences
- [L2] scans a Dockerfile for running as root
- [L2] parses Terraform state and lists resources
- [L2] queries Prometheus for CPU usage over the last hour
- [L3] parses a Windows .evtx event log for failed logins
- [L3] reads a pcap file and extracts DNS queries
- [L3] parses a YARA rule file and lists rule names
- [L3] reads a Kubernetes audit log and finds exec-into-pod events
