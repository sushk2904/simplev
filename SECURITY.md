## 1. Our Threat Model
SimpleV is purposefully designed as a local-first database. By default, it does not listen on any network ports or communicate with external cloud services (aside from the initial download of standard embedding models). Because of this architecture, our primary security focus is on file-system safety and local data integrity rather than network-based attack vectors.

## 2. Security Guidelines
While SimpleV operates locally, we take the safety of your machine seriously. We actively enforce the following precautions:
* **Path Sanitization:** All file paths provided to our document ingestion APIs are strictly sanitized to prevent directory traversal attacks. 
* **Safe Model Execution:** When downloading and executing embedding models via `sentence-transformers`, we prioritize the use of `safetensors` and strictly avoid loading untrusted `pickle` files, protecting your environment from malicious arbitrary code execution.

## 3. Reporting a Vulnerability
We deeply appreciate the community's help in keeping SimpleV secure. If you discover a potential security vulnerability within the project, please do not open a public GitHub issue. 

Instead, we kindly ask that you report it directly to us by emailing `security@simplev.local`. We will review your report promptly, work with you to verify the issue, and coordinate a patch before making the details public.