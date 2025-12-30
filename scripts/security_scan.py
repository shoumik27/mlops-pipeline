"""
Security scanning for dependencies and code vulnerabilities
"""

import subprocess
import json
import os
from datetime import datetime

def run_safety_check():
    """Check for vulnerable dependencies using safety"""
    print("\n" + "="*70)
    print("DEPENDENCY VULNERABILITY SCAN (Safety)")
    print("="*70)
    
    try:
        result = subprocess.run(
            ["safety", "check", "--json"],
            capture_output=True,
            text=True
        )
        
        vulnerabilities = json.loads(result.stdout) if result.stdout else []
        
        if vulnerabilities:
            print(f"⚠️  Found {len(vulnerabilities)} vulnerabilities:")
            for vuln in vulnerabilities:
                print(f"  - {vuln.get('package', 'Unknown')}: {vuln.get('vulnerability', 'Unknown')}")
        else:
            print("✅ No known vulnerabilities found!")
        
        return vulnerabilities
    except Exception as e:
        print(f"❌ Error running safety check: {e}")
        return []

def run_bandit_check():
    """Check for code security issues using bandit"""
    print("\n" + "="*70)
    print("CODE SECURITY SCAN (Bandit)")
    print("="*70)
    
    try:
        result = subprocess.run(
            ["bandit", "-r", "scripts/", "-f", "json"],
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            report = json.loads(result.stdout)
            issues = report.get("results", [])
            
            if issues:
                print(f"⚠️  Found {len(issues)} potential security issues:")
                for issue in issues:
                    print(f"  - {issue.get('issue_severity', 'Unknown')}: {issue.get('issue_text', 'Unknown')}")
            else:
                print("✅ No security issues found!")
            
            return issues
        else:
            print("✅ No security issues found!")
            return []
    except Exception as e:
        print(f"❌ Error running bandit: {e}")
        return []

def generate_sbom():
    """Generate Software Bill of Materials (SBOM)"""
    print("\n" + "="*70)
    print("GENERATING SOFTWARE BILL OF MATERIALS (SBOM)")
    print("="*70)
    
    try:
        result = subprocess.run(
            ["pip", "list", "--format=json"],
            capture_output=True,
            text=True
        )
        
        packages = json.loads(result.stdout)
        
        sbom = {
            "timestamp": datetime.now().isoformat(),
            "package_count": len(packages),
            "packages": packages
        }
        
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/sbom.json", "w") as f:
            json.dump(sbom, f, indent=2)
        
        print(f"✅ SBOM generated with {len(packages)} packages")
        print(f"   Saved to: outputs/sbom.json")
        
        return sbom
    except Exception as e:
        print(f"❌ Error generating SBOM: {e}")
        return {}

def main():
    """Run all security scans"""
    print("\n" + "🔒"*35)
    print("SECURITY SCANNING PIPELINE")
    print("🔒"*35)
    
    security_report = {
        "timestamp": datetime.now().isoformat(),
        "dependency_vulnerabilities": run_safety_check(),
        "code_issues": run_bandit_check(),
        "sbom_generated": True if generate_sbom() else False
    }
    
    # Save report
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/security_report.json", "w") as f:
        json.dump(security_report, f, indent=2)
    
    print("\n" + "="*70)
    print("SECURITY SCAN COMPLETE")
    print("="*70)
    print(f"Report saved to: outputs/security_report.json")
    
    return security_report

if __name__ == "__main__":
    main()
