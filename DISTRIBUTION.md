# Terms for combined desktop builds

The combined ASCII Video program, including the linked PyAV/FFmpeg stack with x264 and x265, is conveyed under the **GNU General Public License, version 3** (`GPL-3.0-only`). Every binary release includes the unmodified GPLv3 text in its license documents. Windows also embeds it under `licenses/` in the executable. The text is available from the [GNU Project](https://www.gnu.org/licenses/gpl-3.0.en.html).

ASCII Video's original source files remain available under the [MIT License](LICENSE). Those files can be used separately under MIT. Selecting GPLv3 for the combined program does not withdraw that permission or replace the licenses of its components.

Third-party code, fonts, and independent runtime components retain their copyright notices, license terms, and applicable exceptions. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the release's `licenses/` directory. This notice does not claim ownership of those components or grant rights beyond the permissions of their respective copyright holders.

## Using and redistributing the program

You may run the program for any purpose, including commercial use, and make private modifications. You may copy, modify, and redistribute the combined program, including for a fee, subject to GPLv3 and the applicable component terms. When conveying a modified version, preserve required notices, identify the changes and their dates, and comply with GPLv3's source and licensing requirements. Do not impose additional restrictions on the rights granted by GPLv3.

The program is supplied without warranty to the extent permitted by applicable law; see sections 15 and 16 of GPLv3 and the component notices. Merely processing your media with ASCII Video does not apply GPLv3 to the resulting videos or images.

This document identifies the licenses and explains the project's release arrangement. The full license texts govern the permissions and obligations.

## Source accompanying binary downloads

For downloads published by this project, Corresponding Source is provided alongside the binary archive, following **GPLv3 section 6(d)**. The binary archive's `README.txt` gives the companion source archive's filename and SHA-256. The download location must offer equivalent access to both archives, with no further charge for the source. Recipients do not have to download the source to obtain the binary.

The source companion must correspond to that exact executable. It includes the application source and the sources of the covered libraries, relevant modifications and patches, and the scripts and instructions needed to build, install, run, and modify the program. The scope and exclusions are those in GPLv3's definition of Corresponding Source; a collection of license texts, a dependency list, or links to current upstream branches does not by itself supply it.

The distributor remains responsible for source availability. Redistributors may use another method permitted by GPLv3 if they satisfy its conditions. This project uses the accompanying-download arrangement rather than relying on a written offer to supply sources later.

## Files in a binary release

- `DISTRIBUTION.md`: this notice for the combined program.
- `COPYING.GPLv3`: the full GNU GPL version 3 text.
- `LICENSE`: the MIT license for the original ASCII Video source files.
- `THIRD_PARTY_NOTICES.md`: component attribution and source-tree notices.
- `licenses/`: the collected component notices and their inventory.
- `README.txt`: the matching source archive's name and checksum.

## Preparing the source review

The Windows packager binds the source review to the final EXE, dependency lock and license manifest. It verifies every embedded DLL, inventoried CPython extension and notice against the reviewed files. The three Microsoft Visual C++ runtime DLLs retain Microsoft's terms. The publisher must establish its entitlement to redistribute the runtime. See [Microsoft's redistribution terms](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files).

On Linux and macOS, finish signing or stapling first when applicable. Run `python scripts/package_posix.py inventory BUNDLE --output SOURCES/bundle-inventory.json` on the build platform. The inventory records the target architecture, lockfile hash, modes, link targets and file hashes. Prepare the corresponding source archives, build instructions and platform-specific notices under `SOURCES/`. Include the project's license documents and `COPYING.GPLv3`.

The file `SOURCES/release-review.json` records the review of that exact artifact. Its `files` object maps every supplied source, notice and inventory path to its SHA-256. The `license_files` list names the documents to distribute with the program. In `components`, each named component lists its native `binaries` from the inventory, its corresponding `sources` and its `notices`. Include sources for code linked statically into those binaries and review bundled Python code, scripts, configuration and data under their respective terms. Record unresolved items in `blockers`. Set `status` to `ready` after completing the review.

Run `python scripts/package_posix.py package BUNDLE --sources SOURCES --name RELEASE` on the target platform to create the binary archive, source archive and checksums together in `dist/RELEASE/`. The corresponding Windows command is documented in [BUILDING.md](BUILDING.md#before-publishing-binaries). The source package must describe the actual release, including any changes made between diagnostic and final builds.
