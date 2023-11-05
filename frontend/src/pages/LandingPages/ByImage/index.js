/*
=========================================================
* Material Kit 2 React - v2.1.0
=========================================================

* Product Page: https://www.creative-tim.com/product/material-kit-react
* Copyright 2023 Creative Tim (https://www.creative-tim.com)

Coded by www.creative-tim.com

 =========================================================

* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

// @mui material components
import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";
import Card from "@mui/material/Card";

// Material Kit 2 React components
import MKBox from "components/MKBox";
// import MKTypography from "components/MKTypography";
// import MKButton from "components/MKButton";

// Material Kit 2 React examples
import DefaultNavbar from "examples/Navbars/DefaultNavbar";
// import DefaultFooter from "examples/Footers/DefaultFooter";

// About Us page sections

// import Information from "pages/LandingPages/ByImage/sections/Information";

// import Team from "pages/LandingPages/ByImage/sections/Team";
// import Featuring from "pages/LandingPages/ByImage/sections/Featuring";
// import Newsletter from "pages/LandingPages/ByImage/sections/Newsletter";

// Routes
import routes from "routes";
import React, { useState } from "react";
import MKTypography from "components/MKTypography";
// import footerRoutes from "footer.routes";

// Images
// import bgImage from "assets/images/bg-about-us.jpg";

function ByImage() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [donenessData, setDonenessData] = useState({ Doneness: "Upload a File" });
  return (
    <>
      <Container>
        <DefaultNavbar routes={routes} />
      </Container>
      <div
        style={{
          backgroundColor: "white",
          height: "100vh",
        }}
      >
        <MKBox component="section" py={12}>
          <Grid
            container
            item
            xs={12}
            lg={7}
            justifyContent="center"
            sx={{ mx: "auto", textAlign: "center" }}
          >
            <h1>Upload Your Steak</h1>
          </Grid>
          <Grid
            container
            item
            spacing={-10}
            xs={12}
            lg={-10}
            justifyContent="center"
            sx={{ mx: "auto", textAlign: "center" }}
          >
            <div style={{ justifyContent: "center" }}>
              {selectedImage && (
                <div>
                  <img alt="not found" width={"500px"} src={URL.createObjectURL(selectedImage)} />
                  <br />
                  <button
                    onClick={() => {
                      setSelectedImage(null);
                      setDonenessData({ Doneness: "Upload a File" });
                    }}
                  >
                    Remove
                  </button>
                </div>
              )}
              <input
                type="file"
                name="myImage"
                onChange={(event) => {
                  console.log(event.target.files[0]);
                  setSelectedImage(event.target.files[0]);
                  const formData = new FormData();
                  formData.append("image", event.target.files[0]);
                  const postRequest = {
                    method: "POST",
                    // headers: { "Content-Type": "image/png" },
                    body: formData,
                  };

                  fetch("http://127.0.0.1:5000/identify", postRequest) //endpoint
                    .then((response) => {
                      return response.json();
                    })
                    .then((data) => {
                      console.log(data);
                      return data;
                    })
                    .then((data) => setDonenessData(data));
                }}
              />
            </div>
            <Card style={{ border: "none", boxShadow: "none" }} sx={{ minWidth: 525 }}>
              <MKTypography variant="h2">Doneness: {donenessData["Doneness"]}</MKTypography>
            </Card>
          </Grid>
        </MKBox>
      </div>
    </>
  );
}

export default ByImage;
