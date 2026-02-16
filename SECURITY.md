# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.4.x   | :white_check_mark: |
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in PyAgent, please report it responsibly.

### How to Report

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Email security concerns to the maintainers directly
3. Include as much detail as possible:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Resolution Timeline**: Depends on severity
  - Critical: 24-72 hours
  - High: 1-2 weeks
  - Medium: 2-4 weeks
  - Low: Next release

### Security Best Practices

When using PyAgent:

1. **Never commit API keys** - Use environment variables
2. **Use Azure AD authentication** - Preferred for enterprise
3. **Validate user inputs** - Before passing to agents
4. **Review agent outputs** - Before executing code
5. **Use sandboxed code execution** - For code generation features

### Dependency Security

We regularly update dependencies to patch known vulnerabilities. Run:

```bash
pip install --upgrade pyagent
```

## Responsible Disclosure

We appreciate the security research community's efforts in keeping PyAgent secure. We will acknowledge researchers who report valid vulnerabilities (with permission).
