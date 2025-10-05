{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d98831e7",
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "import cv2\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Load the image\n",
    "image_path = \"img.jpg\"\n",
    "image = cv2.imread(image_path)\n",
    "gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n",
    "\n",
    "# Threshold\n",
    "_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)\n",
    "\n",
    "# Find all contours with hierarchy\n",
    "contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)\n",
    "\n",
    "# Separate external and internal contours\n",
    "external_contours = []\n",
    "internal_contours = []\n",
    "\n",
    "for i, cnt in enumerate(contours):\n",
    "    if hierarchy[0][i][3] == -1:  # no parent → external\n",
    "        external_contours.append(cnt)\n",
    "    else:  # has a parent → internal\n",
    "        internal_contours.append(cnt)\n",
    "\n",
    "# Draw separately\n",
    "external_img = image.copy()\n",
    "internal_img = image.copy()\n",
    "\n",
    "cv2.drawContours(external_img, external_contours, -1, (0, 255, 0), 2)  # Green\n",
    "cv2.drawContours(internal_img, internal_contours, -1, (0, 0, 255), 2)  # Red\n",
    "\n",
    "# Convert to RGB\n",
    "external_img_rgb = cv2.cvtColor(external_img, cv2.COLOR_BGR2RGB)\n",
    "internal_img_rgb = cv2.cvtColor(internal_img, cv2.COLOR_BGR2RGB)\n",
    "\n",
    "# Show results\n",
    "plt.figure(figsize=(12,6))\n",
    "plt.subplot(1,2,1)\n",
    "plt.imshow(external_img_rgb)\n",
    "plt.title(\"External Contours (Green)\")\n",
    "plt.axis(\"off\")\n",
    "\n",
    "plt.subplot(1,2,2)\n",
    "plt.imshow(internal_img_rgb)\n",
    "plt.title(\"Internal Contours (Red)\")\n",
    "plt.axis(\"off\")\n",
    "\n",
    "plt.show()\n"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
