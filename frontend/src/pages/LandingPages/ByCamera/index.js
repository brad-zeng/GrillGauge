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
import Grid from "@mui/material/Grid";

// // Material Kit 2 React components
// import MKBox from "components/MKBox";
// import MKInput from "components/MKInput";
// import MKButton from "components/MKButton";
import MKTypography from "components/MKTypography";

// // Material Kit 2 React examples
import DefaultNavbar from "examples/Navbars/DefaultNavbar";
// import DefaultFooter from "examples/Footers/DefaultFooter";

// // Routes
import routes from "routes";
// import footerRoutes from "footer.routes";

// // Image
// import bgImage from "assets/images/illustrations/illustration-reset.jpg";

import Container from "@mui/material/Container";
import React, { useState, useRef, useEffect } from "react";

function ByCamera() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  // const [pictureData, setPictureData] = useState(null);
  const [donenessData, setDonenessData] = useState({ Doneness: "" });

  useEffect(() => {
    const startWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoRef.current.srcObject = stream;
      } catch (error) {
        console.error("Error accessing webcam:", error);
      }
    };

    startWebcam();
  }, []);

  const capturePicture = () => {
    if (videoRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const context = canvas.getContext("2d");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      const dataUrl = canvas.toDataURL("image/png");
      // setPictureData(dataUrl);
      console.log(dataUrl);
      const postRequest = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataUrl }),
      };

      fetch("http://127.0.0.1:5000/identifyJson", postRequest) //endpoint
        .then((response) => {
          return response.json();
        })
        .then((data) => {
          console.log(data);
          return data;
        })
        .then((data) => setDonenessData(data));
    }
  };

  useEffect(() => {
    const intervalId = setInterval(capturePicture, 2000); // Capture a picture every 2 seconds

    return () => {
      clearInterval(intervalId); // Cleanup the interval when the component unmounts
    };
  }, []);

  return (
    <>
      <Container>
        <DefaultNavbar routes={routes} />
      </Container>
      <Container>
        <Grid
          container
          spacing={3}
          item
          padding="125px"
          xs={12}
          lg={7}
          justifyContent="center"
          sx={{ mx: "auto", textAlign: "center" }}
        >
          <div>
            <Grid
              container
              spacing={3}
              item
              xs={12}
              lg={7}
              justifyContent="center"
              sx={{ mx: "auto", textAlign: "center" }}
            >
              <video ref={videoRef} autoPlay playsInline muted />
            </Grid>
            <canvas ref={canvasRef} style={{ display: "none" }} />
            {/* {pictureData && <img src={pictureData} alt="Webcam Capture" />} */}
            <MKTypography variant="h2">Doneness: {donenessData["Doneness"]}</MKTypography>
          </div>
        </Grid>
      </Container>
    </>
  );
}

export default ByCamera;

/*
function ByCamera() {
  return (
    <>
      <MKBox position="fixed" top="0.5rem" width="100%">
        <DefaultNavbar routes={routes} />
      </MKBox>
      <Grid container spacing={3} alignItems="center">
        <Grid item xs={12} lg={6}>
          <MKBox
            display={{ xs: "none", lg: "flex" }}
            width="calc(100% - 2rem)"
            height="calc(100vh - 2rem)"
            borderRadius="lg"
            ml={2}
            mt={2}
            sx={{ backgroundImage: `url(${bgImage})` }}
          />
        </Grid>
        <Grid
          item
          xs={12}
          sm={10}
          md={7}
          lg={6}
          xl={4}
          ml={{ xs: "auto", lg: 6 }}
          mr={{ xs: "auto", lg: 6 }}
        >
          <MKBox
            bgColor="white"
            borderRadius="xl"
            shadow="lg"
            display="flex"
            flexDirection="column"
            justifyContent="center"
            mt={{ xs: 20, sm: 18, md: 20 }}
            mb={{ xs: 20, sm: 18, md: 20 }}
            mx={3}
          >
            <MKBox
              variant="gradient"
              bgColor="info"
              coloredShadow="info"
              borderRadius="lg"
              p={2}
              mx={2}
              mt={-3}
            >
              <MKTypography variant="h3" color="white">
                Contact us
              </MKTypography>
            </MKBox>
            <MKBox p={3}>
              <MKTypography variant="body2" color="text" mb={3}>
                For further questions, including partnership opportunities, please email
                hello@creative-tim.com or contact using our contact form.
              </MKTypography>
              <MKBox width="100%" component="form" method="post" autoComplete="off">
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <MKInput
                      variant="standard"
                      label="Full Name"
                      InputLabelProps={{ shrink: true }}
                      fullWidth
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <MKInput
                      type="email"
                      variant="standard"
                      label="Email"
                      InputLabelProps={{ shrink: true }}
                      fullWidth
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <MKInput
                      variant="standard"
                      label="What can we help you?"
                      placeholder="Describe your problem in at least 250 characters"
                      InputLabelProps={{ shrink: true }}
                      multiline
                      fullWidth
                      rows={6}
                    />
                  </Grid>
                </Grid>
                <Grid container item justifyContent="center" xs={12} mt={5} mb={2}>
                  <MKButton type="submit" variant="gradient" color="info">
                    Send Message
                  </MKButton>
                </Grid>
              </MKBox>
            </MKBox>
          </MKBox>
        </Grid>
      </Grid>
      <MKBox pt={6} px={1} mt={6}>
        <DefaultFooter content={footerRoutes} />
      </MKBox>
    </>
  );
}

 export default ByCamera; */
