# MCP Security Checklist

A comprehensive security checklist for deploying production MCP servers.

## Pre-Deployment Security

### Authentication & Authorization

- [ ] **Multi-factor authentication (MFA)** enabled for all users
- [ ] **OAuth 2.0** properly configured with secure redirect URIs
- [ ] **API keys** stored in secure vault (HashiCorp Vault, AWS Secrets Manager)
- [ ] **JWT tokens** use strong signing algorithms (RS256, ES256)
- [ ] **Token expiration** set appropriately (access: 15min, refresh: 7 days)
- [ ] **Session management** with secure, httpOnly cookies
- [ ] **Password policies** enforce complexity requirements
- [ ] **Account lockout** after failed login attempts
- [ ] **SSO integration** tested and working
- [ ] **Service accounts** use separate credentials

### Network Security

- [ ] **TLS 1.3** (minimum TLS 1.2) for all connections
- [ ] **Certificate validation** enabled and tested
- [ ] **Certificate pinning** for critical connections
- [ ] **Firewall rules** restrict access to necessary ports only
- [ ] **VPN/Private network** for internal communications
- [ ] **DDoS protection** configured (Cloudflare, AWS Shield)
- [ ] **Rate limiting** implemented (per user, per IP)
- [ ] **IP whitelisting** for administrative access
- [ ] **Network segmentation** isolates MCP servers
- [ ] **Load balancer** configured with security headers

### Data Protection

- [ ] **Encryption at rest** for all sensitive data
- [ ] **Encryption in transit** (TLS) for all communications
- [ ] **Database encryption** enabled
- [ ] **Backup encryption** configured
- [ ] **PII data masking** in logs and responses
- [ ] **Data retention policies** defined and enforced
- [ ] **Secure deletion** procedures for sensitive data
- [ ] **Key rotation** schedule established
- [ ] **Secrets management** using vault solution
- [ ] **Environment variables** never contain secrets

### Input Validation

- [ ] **SQL injection** prevention (parameterized queries)
- [ ] **XSS prevention** (input sanitization, output encoding)
- [ ] **CSRF protection** enabled
- [ ] **Command injection** prevention
- [ ] **Path traversal** prevention
- [ ] **File upload** validation (type, size, content)
- [ ] **JSON schema** validation for all inputs
- [ ] **Request size limits** enforced
- [ ] **Content-Type** validation
- [ ] **Regex DoS** prevention (timeout limits)

### Code Security

- [ ] **Dependency scanning** (Snyk, Dependabot)
- [ ] **No hardcoded secrets** in code
- [ ] **Security linting** (Bandit for Python, ESLint)
- [ ] **SAST tools** integrated in CI/CD
- [ ] **Code review** process includes security check
- [ ] **Third-party libraries** vetted and up-to-date
- [ ] **Minimal dependencies** principle followed
- [ ] **License compliance** verified
- [ ] **Vulnerability scanning** automated
- [ ] **Security patches** applied promptly

## Deployment Security

### Infrastructure

- [ ] **Principle of least privilege** for all services
- [ ] **Container security** (non-root user, minimal base image)
- [ ] **Kubernetes security** (RBAC, Pod Security Policies)
- [ ] **Cloud security groups** properly configured
- [ ] **IAM roles** follow least privilege
- [ ] **Resource limits** set (CPU, memory, connections)
- [ ] **Auto-scaling** configured with limits
- [ ] **Health checks** implemented
- [ ] **Graceful shutdown** handling
- [ ] **Immutable infrastructure** where possible

### Monitoring & Logging

- [ ] **Audit logging** for all security events
- [ ] **Centralized logging** (ELK, Splunk, CloudWatch)
- [ ] **Log retention** meets compliance requirements
- [ ] **Log integrity** protected (immutable storage)
- [ ] **Security alerts** configured
- [ ] **Anomaly detection** enabled
- [ ] **Failed login monitoring** with alerts
- [ ] **Privilege escalation** monitoring
- [ ] **Data access logging** comprehensive
- [ ] **Log sanitization** removes sensitive data

### Compliance

- [ ] **SOC 2** requirements met (if applicable)
- [ ] **GDPR** compliance verified (if applicable)
- [ ] **HIPAA** compliance verified (if applicable)
- [ ] **PCI DSS** compliance verified (if applicable)
- [ ] **Data residency** requirements met
- [ ] **Privacy policy** updated
- [ ] **Terms of service** reviewed
- [ ] **Consent management** implemented
- [ ] **Right to deletion** process defined
- [ ] **Data breach** response plan documented

## Runtime Security

### Access Control

- [ ] **RBAC** (Role-Based Access Control) implemented
- [ ] **ABAC** (Attribute-Based Access Control) if needed
- [ ] **Tool-level permissions** enforced
- [ ] **Resource-level permissions** enforced
- [ ] **Time-based access** controls if needed
- [ ] **Separation of duties** enforced
- [ ] **Admin access** requires additional authentication
- [ ] **Service-to-service** authentication required
- [ ] **Permission audits** scheduled regularly
- [ ] **Access reviews** conducted quarterly

### Error Handling

- [ ] **Generic error messages** to users (no stack traces)
- [ ] **Detailed errors** logged securely
- [ ] **Error rate monitoring** with alerts
- [ ] **Circuit breakers** prevent cascade failures
- [ ] **Timeout handling** prevents resource exhaustion
- [ ] **Graceful degradation** for dependencies
- [ ] **Retry logic** with exponential backoff
- [ ] **Dead letter queues** for failed operations
- [ ] **Error correlation IDs** for debugging
- [ ] **Security errors** trigger alerts

### API Security

- [ ] **API versioning** implemented
- [ ] **Backward compatibility** maintained
- [ ] **Deprecation notices** provided
- [ ] **API documentation** accurate and current
- [ ] **OpenAPI/Swagger** spec available
- [ ] **API gateway** configured
- [ ] **Request throttling** per endpoint
- [ ] **CORS** properly configured
- [ ] **Content Security Policy** headers set
- [ ] **Security headers** (HSTS, X-Frame-Options, etc.)

## Incident Response

### Preparation

- [ ] **Incident response plan** documented
- [ ] **Security team** contacts listed
- [ ] **Escalation procedures** defined
- [ ] **Communication templates** prepared
- [ ] **Backup contacts** identified
- [ ] **Legal counsel** contact available
- [ ] **PR team** contact available
- [ ] **Runbooks** for common incidents
- [ ] **Disaster recovery** plan tested
- [ ] **Business continuity** plan documented

### Detection

- [ ] **Intrusion detection** system configured
- [ ] **Security monitoring** 24/7
- [ ] **Alert thresholds** properly tuned
- [ ] **Anomaly detection** baseline established
- [ ] **Threat intelligence** feeds integrated
- [ ] **Vulnerability scanning** automated
- [ ] **Penetration testing** scheduled
- [ ] **Bug bounty program** considered
- [ ] **Security metrics** tracked
- [ ] **Incident tracking** system in place

### Response

- [ ] **Incident classification** criteria defined
- [ ] **Response procedures** documented
- [ ] **Evidence preservation** procedures
- [ ] **Forensics tools** available
- [ ] **Containment strategies** defined
- [ ] **Eradication procedures** documented
- [ ] **Recovery procedures** tested
- [ ] **Post-incident review** process
- [ ] **Lessons learned** documentation
- [ ] **Improvement actions** tracked

## Ongoing Security

### Regular Activities

- [ ] **Security audits** quarterly
- [ ] **Penetration testing** annually
- [ ] **Vulnerability assessments** monthly
- [ ] **Access reviews** quarterly
- [ ] **Security training** for all team members
- [ ] **Phishing simulations** quarterly
- [ ] **Disaster recovery drills** semi-annually
- [ ] **Backup testing** monthly
- [ ] **Security metrics review** monthly
- [ ] **Compliance audits** as required

### Updates & Patches

- [ ] **Security patch** process defined
- [ ] **Critical patches** applied within 24 hours
- [ ] **Regular patches** applied within 7 days
- [ ] **Dependency updates** automated
- [ ] **OS updates** scheduled
- [ ] **Container image** updates automated
- [ ] **Certificate renewal** automated
- [ ] **Key rotation** automated
- [ ] **Rollback procedures** tested
- [ ] **Change management** process followed

### Documentation

- [ ] **Security architecture** documented
- [ ] **Threat model** documented
- [ ] **Security controls** documented
- [ ] **Incident response** plan current
- [ ] **Runbooks** up to date
- [ ] **Security policies** published
- [ ] **Training materials** current
- [ ] **Audit reports** archived
- [ ] **Compliance documentation** current
- [ ] **Security roadmap** defined

## Tool-Specific Security

### Database Tools

- [ ] **Parameterized queries** only
- [ ] **Connection pooling** with limits
- [ ] **Read-only** connections where possible
- [ ] **Query timeouts** enforced
- [ ] **Result set limits** enforced
- [ ] **Database credentials** rotated regularly
- [ ] **Database encryption** enabled
- [ ] **Audit logging** for all queries
- [ ] **Row-level security** if applicable
- [ ] **Database firewall** rules configured

### API Integration Tools

- [ ] **API credentials** in vault
- [ ] **OAuth tokens** refreshed automatically
- [ ] **API rate limits** respected
- [ ] **Retry logic** with backoff
- [ ] **Circuit breakers** for external APIs
- [ ] **Request/response** logging
- [ ] **Timeout handling** implemented
- [ ] **SSL/TLS** verification enabled
- [ ] **API versioning** handled
- [ ] **Webhook signatures** verified

### File Operations Tools

- [ ] **Path validation** prevents traversal
- [ ] **File type** validation
- [ ] **File size** limits enforced
- [ ] **Virus scanning** for uploads
- [ ] **Temporary files** cleaned up
- [ ] **File permissions** restrictive
- [ ] **Storage quotas** enforced
- [ ] **Encryption** for sensitive files
- [ ] **Access logging** for file operations
- [ ] **Retention policies** enforced

## Compliance Checklists

### SOC 2

- [ ] Access controls documented
- [ ] Change management process
- [ ] Incident response procedures
- [ ] Monitoring and logging
- [ ] Vendor management
- [ ] Business continuity plan
- [ ] Risk assessment completed
- [ ] Security awareness training
- [ ] Audit trail integrity
- [ ] Annual audit scheduled

### GDPR

- [ ] Data inventory completed
- [ ] Privacy policy updated
- [ ] Consent management
- [ ] Right to access implemented
- [ ] Right to deletion implemented
- [ ] Data portability supported
- [ ] Breach notification process
- [ ] DPO appointed (if required)
- [ ] Data processing agreements
- [ ] Privacy impact assessments

### HIPAA

- [ ] PHI identified and protected
- [ ] Access controls implemented
- [ ] Audit controls enabled
- [ ] Integrity controls in place
- [ ] Transmission security
- [ ] Business associate agreements
- [ ] Breach notification procedures
- [ ] Security risk analysis
- [ ] Workforce training
- [ ] Contingency planning

## Sign-Off

**Security Review Completed By**: ___________________  
**Date**: ___________________  
**Next Review Date**: ___________________  

**Approved By**: ___________________  
**Date**: ___________________  

---

**Remember**: Security is not a one-time activity. Regular reviews and updates are essential!

**Last Updated**: January 2025
