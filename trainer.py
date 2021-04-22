"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/GameGAN_code.
Authors: Seung Wook Kim, Yuhao Zhou, Jonah Philion, Antonio Torralba, Sanja Fidler
"""

"""
Contains some code from:
https://github.com/LMescheder/GAN_stability
with the following license:

MIT License

Copyright (c) 2018 Lars Mescheder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import utils
import torch
import torch.nn.functional as F
import torch.utils.data
import math


class Trainer(object):
    def __init__(self, opts,
                 netG, netD,
                 optG_temporal, optG_graphic, optD,
                 gan_type, reg_type, reg_param, zdist):

        self.opts = opts

        self.netG = netG
        self.netG.opts = opts
        self.netD = netD
        if self.netD is not None:
            self.netD.opts = opts

        self.optG_temporal = optG_temporal
        self.optG_graphic = optG_graphic
        self.optD = optD

        self.gan_type = gan_type
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.zdist = zdist

        # Default to hinge loss
        if utils.check_arg(opts, 'standard_gan_loss'):
            self.generator_loss = self.standard_gan_loss
            self.discriminator_loss = self.standard_gan_loss
        else:
            self.generator_loss = self.loss_hinge_gen
            self.discriminator_loss = self.loss_hinge_dis

    # Hinge loss for discriminator
    def loss_hinge_dis(self, logits, label):
        if label == 1:
            return torch.mean(F.relu(1. - logits))
        else:
            return torch.mean(F.relu(1. + logits))

    # Hinge loss for generator
    def loss_hinge_gen(self, dis_fake):
        loss = -torch.mean(dis_fake)
        return loss

    # BCE GAN loss
    def standard_gan_loss(self, d_out, target=1):
        if d_out is None:
            return utils.check_gpu(self.opts.gpu, torch.FloatTensor([0]))
        targets = d_out.new_full(size=d_out.size(), fill_value=target)

        loss = F.binary_cross_entropy_with_logits(d_out, targets)
        return loss

    # Reconstruction loss
    def get_recon_loss(self, input, target, detach=True, criterion=None):

        if detach:
            target = target.detach()
        loss = criterion(input, target, reduction='sum') / target.size(0)
        return loss

    def generator_trainstep(self, states, actions, warm_up=10, train=True, epoch=0):
        '''
        Single run of episode training / inference
        '''
        if self.opts.warmup_decay_epoch > 0:
            warm_up = max(self.opts.min_warmup, math.ceil(warm_up * (1 - epoch * 1.0 / self.opts.warmup_decay_epoch)))


        utils.toggle_grad(self.netG, True)
        utils.toggle_grad(self.netD, True)
        if train:
            self.netG.train()
            self.netD.train()
        else:
            self.netG.eval()
            self.netD.eval()

        self.optD.zero_grad()
        self.optG_temporal.zero_grad()
        self.optG_graphic.zero_grad()

        loss_dict, grads = {}, {}
        gen_actions = actions

        graphic_loss, temporal_loss, total_loss, dout_fake, rev_images = 0, 0, 0, None, None

        # generate outputs
        gout = self.netG(self.zdist, states, gen_actions, warm_up, train=train, epoch=epoch)

        # adversarial losses
        if self.opts.gan_loss:
            # run discriminators
            gen_adv_input = torch.cat(gout['outputs'], dim=0)
            dout_fake = self.netD(gen_adv_input, gen_actions[:len(gout['outputs']) + 1], states, warm_up, epoch=epoch)

            # action-conditioned discriminator loss
            gloss_fake_action = self.generator_loss(dout_fake['action_predictions'])
            loss_dict['gloss_fake_action'] = gloss_fake_action

            # single frame discrimiinator loss
            gloss_single_frame_loss = self.generator_loss(dout_fake['single_frame_predictions_all'])
            if dout_fake['single_frame_predictions_patch'] is not None:
                gloss_single_frame_patch_loss = self.generator_loss(dout_fake['single_frame_predictions_patch'])
                gloss_single_frame_loss += gloss_single_frame_patch_loss
                loss_dict['gloss_single_frame_patch_loss'] = gloss_single_frame_patch_loss
            loss_dict['gloss_single_frame_loss'] = gloss_single_frame_loss

            # temporal discriminator loss
            gloss_content_loss = 0
            if self.opts.do_temporal:
                for i in range(len(dout_fake['content_predictions'])):
                    curloss = self.generator_loss(dout_fake['content_predictions'][i])
                    loss_dict['gloss_content_loss' + str(i)] = curloss
                    gloss_content_loss += curloss
                gloss_content_loss = gloss_content_loss / len(dout_fake['content_predictions'])

            # action and z reconstruction losses
            action_orig = torch.cat(gen_actions[:len(gout['outputs'])], dim=0)
            _, action_orig = torch.max(action_orig, 1)
            action_orig = action_orig.long()
            z_orig = torch.cat(gout['zs'], dim=0)
            action_recon_loss = F.cross_entropy(dout_fake['action_recon'], action_orig)
            z_recon_loss = F.mse_loss(dout_fake['z_recon'], z_orig)
            loss_dict['g_z_recon_loss'] = z_recon_loss
            loss_dict['g_action_recon_loss'] = action_recon_loss

            # some all adversarial losses
            total_loss += self.opts.gen_content_loss_multiplier * gloss_content_loss + \
                          self.opts.gen_single_frame_loss_multiplier * gloss_single_frame_loss + \
                          self.opts.gen_action_loss_multiplier * gloss_fake_action + \
                          self.opts.gen_info_loss_multiplier * z_recon_loss + \
                          action_recon_loss

            # discriminator feature matching losses
            if self.opts.feature_l2:
                feat_loss_fn = F.mse_loss
            else:
                feat_loss_fn = F.l1_loss
            din = states[1:len(gout['outputs']) + 1]
            dout_real = self.netD(torch.cat(din, dim=0), actions[:len(gout['outputs']) + 1], states, warm_up, epoch=epoch)
            if self.opts.disc_features:
                x_fake_ = dout_fake['disc_features']
                x_real_ = dout_real['disc_features'].detach()
                loss_l1_disc_features = feat_loss_fn(x_fake_, x_real_)
                loss_dict['loss_l1_disc_features'] = loss_l1_disc_features
                total_loss += self.opts.feature_loss_multiplier * (loss_l1_disc_features)
            if self.opts.disc_temporal_features:
                x_fake_ = dout_fake['disc_temporal_features']
                x_real_ = dout_real['disc_temporal_features'].detach()
                loss_l1_disc_temporal_features = feat_loss_fn(x_fake_, x_real_)
                loss_dict['loss_l1_disc_temporal_features'] = loss_l1_disc_temporal_features
                total_loss += self.opts.feature_loss_multiplier * (loss_l1_disc_temporal_features)

        # frame reconstruction loss
        targ = torch.cat(states[1:len(gout['outputs']) + 1], dim=0)
        x_fake_ = torch.cat(gout['outputs'], dim=0)
        x_real_ = targ
        if self.opts.final_l1:
            criterion = F.l1_loss
        elif self.opts.final_l2:
            criterion = F.mse_loss
        loss_recon = self.get_recon_loss(x_fake_, x_real_, criterion=criterion)
        total_loss += self.opts.recon_loss_multiplier * loss_recon

        loss_dict['loss_recon'] = loss_recon

        # cycle_loss
        if self.opts.do_memory and (self.opts.cycle_loss and epoch >= self.opts.cycle_start_epoch):
            if self.opts.rev_multiply_map:
                ref = [comp[0] for comp in gout['base_imgs_all']]
            else:
                ref = [comp[2] for comp in gout['base_imgs_all']]

            rev_outputs = torch.cat(gout['rev_outputs'][::-1], dim=0)
            num_rev = len(gout['rev_outputs'])
            gout['rev_inputs'] = ref[:num_rev]
            rev_reference = torch.cat(gout['rev_inputs'], dim=0)
            if self.opts.rev_final_l1:
                criterion = F.l1_loss
            elif self.opts.rev_final_l2:
                criterion = F.mse_loss
            loss_rev_recon = self.get_recon_loss(rev_outputs, rev_reference, detach=False,
                                                 criterion=criterion)
            cycle_loss = self.opts.rev_recon_loss_multiplier * loss_rev_recon
            loss_dict['loss_rev_recon'] = loss_rev_recon

        # memory regularization
        if self.opts.do_memory and self.opts.alpha_loss_multiplier > 0:
            total_loss += self.opts.alpha_loss_multiplier * gout['alpha_loss']
            loss_dict['loss_alpha'] = gout['alpha_loss']

        # optimization
        if train:
            if self.opts.gan_loss:
                gen_adv_input.register_hook(utils.save_grad('gen_adv_input', grads))
            x_fake_.register_hook(utils.save_grad('gen_recon_input', grads))

            if self.opts.do_memory and self.opts.cycle_loss and epoch >= self.opts.cycle_start_epoch:

                '''
                # gradient from cycle loss only applied to dynamics engine and memory
                (total_loss+cycle_loss).backward(retain_graph=True)
                self.optG_temporal.step()

                self.optG_temporal.zero_grad()
                self.optG_graphic.zero_grad()
                total_loss.backward()
                self.optG_graphic.step()
                '''

                # With Torch 1.5+ there is a fix to the checks in the autograd and above code yields an error:
                ## RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation:
                ## [torch.cuda.FloatTensor [512, 1536]], which is output 0 of TBackward, "is at version 2; expected version 1 instead".
                ## Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
                # What this does mean is that once we step the optG_temporal, it updates parameters in-place and the total_loss.backward()
                # cannot be computed anymore. Previously, the error did not get raised due to a bug.
                # The fix that we are going to perform here is to calculate gradients of the (total_loss+cycle_loss),
                # save them aside filtered by the parameters (leafs) optimized by the optG_temporal optimizer,
                # zero gradients and calculate gradients for total_loss's backward step, then set the previously saved aside back
                # and step both optimizers after all of this.
                # Is there a better way to do it without rewriting a lot of the code here?

                # gradient from cycle loss only applied to dynamics engine and memory
                (total_loss+cycle_loss).backward(retain_graph=True)

                # Save gradients aside
                saved_grad_groups = []
                for param_group in self.optG_temporal.param_groups:
                    saved_grad_groups.append([])
                    for params in param_group['params']:
                        saved_grad_groups[-1].append(params.grad.clone())

                # Zero gradients
                self.optG_temporal.zero_grad()
                self.optG_graphic.zero_grad()

                # Calculate gradients from total_loss alone
                total_loss.backward()

                # Set the gradients of combined loss back
                for param_group, saved_grads in zip(self.optG_temporal.param_groups, saved_grad_groups):
                    for params, saved_grad in zip(param_group['params'], saved_grads):
                        params.grad.detach()
                        del params.grad
                        params.grad = saved_grad

                torch.cuda.empty_cache()

                # Optimize
                self.optG_temporal.step()
                self.optG_graphic.step()

            else:
                total_loss.backward()
                self.optG_temporal.step()
                if not self.opts.fix_graphic:
                    self.optG_graphic.step()
        return loss_dict, total_loss, gout, grads, None

    def discriminator_trainstep(self, states, actions, neg_actions, warm_up=10, gout=None, dout_fake=None,
                                epoch=0, step=0):
        '''
        Single step of discriminator training
        '''
        if self.opts.warmup_decay_epoch > 0:
            warm_up = max(self.opts.min_warmup, math.ceil(warm_up * (1 - epoch * 1.0 / self.opts.warmup_decay_epoch)))

        utils.toggle_grad(self.netG, False)
        utils.toggle_grad(self.netD, True)
        self.netG.train()
        self.netD.train()
        self.optG_temporal.zero_grad()
        self.optG_graphic.zero_grad()

        self.optD.zero_grad()

        loss_dict = {}


        states = [x.requires_grad_() for x in states]
        actions = [x.requires_grad_() for x in actions]
        neg_actions = [x.requires_grad_() for x in neg_actions]

        # Run discriminators on real data
        d_input = torch.cat(states[1:], dim=0)
        d_input = d_input.requires_grad_()
        dout = self.netD(d_input, actions, states, warm_up, neg_actions=neg_actions,
                         rev_steps=-1, epoch=epoch, step=step)

        # action-conditioned disc loss for real and negative actions
        dloss_real_action = self.discriminator_loss(dout['action_predictions'], 1)
        dloss_real_action_wrong = self.discriminator_loss(dout['neg_action_predictions'], 0)
        loss_dict['dloss_real_action'] = dloss_real_action
        loss_dict['dloss_real_action_wrong'] = dloss_real_action_wrong

        # action reconstruction loss from real data
        action_orig = torch.cat(actions[:-1], dim=0)
        _, action_orig = torch.max(action_orig, 1)
        action_orig = action_orig.long()
        action_recon_loss = F.cross_entropy(dout['action_recon'], action_orig)
        loss_dict['d_action_recon_loss'] = action_recon_loss

        # single frame disc loss
        dloss_real_single_frame_loss = self.discriminator_loss(dout['single_frame_predictions_all'], 1)
        if dout['single_frame_predictions_patch'] is not None:
            dloss_real_single_frame_patch_loss = self.discriminator_loss(dout['single_frame_predictions_patch'], 1)
            dloss_real_single_frame_loss += dloss_real_single_frame_patch_loss
            loss_dict['dloss_real_single_frame_patch_loss'] = dloss_real_single_frame_patch_loss
        loss_dict['dloss_real_single_frame_loss'] = dloss_real_single_frame_loss

        # temporal disc loss
        dloss_real_content_loss = 0
        if self.opts.do_temporal:
            for i in range(len(dout['content_predictions'])):
                curloss = self.discriminator_loss(dout['content_predictions'][i], 1)
                loss_dict['dloss_real_content_loss' + str(i)] = curloss
                dloss_real_content_loss += curloss
            dloss_real_content_loss = dloss_real_content_loss / len(dout['content_predictions'])
            loss_dict['dloss_real_content_loss'] = dloss_real_content_loss


        loss = self.opts.disc_loss_multiplier * (self.opts.disc_content_loss_multiplier * dloss_real_content_loss + \
                                                  dloss_real_single_frame_loss) + \
                                                  dloss_real_action + dloss_real_action_wrong + \
                                                  action_recon_loss

        # gradient penalty on real data
        reg = 0
        if (self.reg_type == 'real' or self.reg_type == 'real_fake') and self.reg_param > 0:
            reg += 0.33*utils.compute_grad2(dout['action_predictions'], d_input, ns=self.opts.num_steps).mean()
            reg += 0.33*utils.compute_grad2(dout['single_frame_predictions_all'], d_input, ns=self.opts.num_steps).mean()
            reg += 0.33*utils.compute_grad2(dout['action_recon'], d_input, ns=self.opts.num_steps).mean()
            reg_temporal = 0
            if self.opts.do_temporal:
                for i in range(len(dout['content_predictions'])):
                    curloss = utils.compute_grad2(dout['content_predictions'][i], d_input, ns=self.opts.num_steps).mean()
                    reg_temporal += curloss
                reg_temporal = reg_temporal / len(dout['content_predictions'])
                loss_dict['dloss_REG_temporal'] = reg_temporal

            loss_dict['dloss_REG'] = reg
            loss += self.reg_param * reg + self.opts.LAMBDA_temporal * reg_temporal

        # Run discriminators on generated data
        if dout_fake is None:
            gen_actions = actions
            dout_fake = self.netD(torch.cat(gout['outputs'], dim=0).detach(), gen_actions[:len(gout['outputs']) + 1],
                                  states, warm_up, rev_steps=-1, epoch=epoch)

        # action-conditioned disc loss on generated data
        dloss_fake_action = self.discriminator_loss(dout_fake['action_predictions'], 0)
        loss_dict['dloss_fake_action'] = dloss_fake_action

        # single frame disc loss on generated data
        dloss_fake_single_frame_loss = self.discriminator_loss(dout_fake['single_frame_predictions_all'], 0)
        if dout_fake['single_frame_predictions_patch'] is not None:
            dloss_fake_single_frame_patch_loss = self.discriminator_loss(dout_fake['single_frame_predictions_patch'], 0)
            dloss_fake_single_frame_loss += dloss_fake_single_frame_patch_loss
            loss_dict['dloss_fake_single_frame_patch_loss'] = dloss_fake_single_frame_patch_loss
        loss_dict['dloss_fake_single_frame_loss'] = dloss_fake_single_frame_loss

        # temporal disc loss on generated data
        dloss_fake_content_loss = 0
        if self.opts.do_temporal:
            for i in range(len(dout_fake['content_predictions'])):
                curloss = self.discriminator_loss(dout_fake['content_predictions'][i], 0)
                loss_dict['dloss_fake_content_loss' + str(i)] = curloss
                dloss_fake_content_loss += curloss
            dloss_fake_content_loss = dloss_fake_content_loss / len(dout_fake['content_predictions'])
            loss_dict['dloss_fake_content_loss'] = dloss_fake_content_loss

        # action and z reconstruction losses
        z_orig = torch.cat(gout['zs'], dim=0)
        z_recon_loss = F.mse_loss(dout_fake['z_recon'], z_orig)  # , size_average=True) / z_orig.size(0)
        loss_dict['dloss_fake_z_recon_loss'] = z_recon_loss

        loss += self.opts.disc_loss_multiplier * (self.opts.disc_content_loss_multiplier * dloss_fake_content_loss + \
                                                  dloss_fake_single_frame_loss) + \
                                                  dloss_fake_action + \
                                                  z_recon_loss
        loss.backward()
        self.optD.step()
        utils.toggle_grad(self.netD, False)

        return loss_dict
