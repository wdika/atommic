# coding=utf-8

MAJOR = 1
MINOR = 0
PATCH = 0
PRE_RELEASE = ""

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = ".".join(map(str, VERSION[:3]))
__version__ = ".".join(map(str, VERSION[:3])) + "".join(VERSION[3:])

__package_name__ = "atommic"
__contact_names__ = "Dimitris Karkalousos"
__contact_emails__ = "d.karkalousos@amsterdamumc.nl"
__homepage__ = "https://github.com/wdika/atommic"
__repository_url__ = "https://github.com/wdika/atommic"
__download_url__ = "https://github.com/wdika/atommic/releases"
__description__ = "Advanced Toolbox for Multitask Medical Imaging Consistency (ATOMMIC)"
__license__ = "Apache-2.0 License"
__keywords__ = (
    "deep-learning, medical-imaging, mri, quantitative-imaging, medical-image-processing, medical-image-analysis"
)
